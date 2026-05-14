import os
import sys
import json
import uuid
import queue
import shutil
import tempfile
import threading
import subprocess
from pathlib import Path
from datetime import timedelta

from flask import Flask, request, jsonify, Response, render_template, stream_with_context, send_file

STATUS_QUEUED = "queued"
STATUS_EXTRACTING = "extracting_audio"
STATUS_LOADING = "loading_model"
STATUS_TRANSCRIBING = "transcribing"
STATUS_DONE = "done"
STATUS_ERROR = "error"
STATUS_DIARIZING = "diarizing"

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
app = Flask(__name__)

tasks: dict[str, dict] = {}
tasks_lock = threading.Lock()

MODELS = {"medium", "large-v3"}

TEMP_ROOT = Path(tempfile.gettempdir()) / "whisper_web"
TEMP_ROOT.mkdir(exist_ok=True)

_FFMPEG_AVAILABLE: bool | None = None
_MODEL_STATUS_CACHE: dict | None = None


def _find_ffmpeg() -> str:
    if shutil.which("ffmpeg"):
        return "ffmpeg"

    winget_base = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    if winget_base.exists():
        for p in winget_base.glob("Gyan.FFmpeg_*"):
            for exe in p.glob("**/ffmpeg.exe"):
                return str(exe)

    for p in [
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
    ]:
        if os.path.exists(p):
            return p

    return "ffmpeg"


FFMPEG_PATH = _find_ffmpeg()


def check_ffmpeg() -> bool:
    global _FFMPEG_AVAILABLE
    if _FFMPEG_AVAILABLE is not None:
        return _FFMPEG_AVAILABLE
    try:
        result = subprocess.run(
            [FFMPEG_PATH, "-version"],
            capture_output=True,
            timeout=10,
        )
        _FFMPEG_AVAILABLE = result.returncode == 0
    except Exception:
        _FFMPEG_AVAILABLE = False
    return _FFMPEG_AVAILABLE


def extract_audio(video_path: str, output_dir: str, task_id: str) -> str:
    audio_path = os.path.join(output_dir, f"{task_id}_audio.wav")
    cmd = [
        FFMPEG_PATH, "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_path,
    ]
    proc = subprocess.Popen(
        cmd,
        stderr=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        universal_newlines=True,
    )
    _, stderr = proc.communicate(timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 音频提取失败: {stderr[-500:]}")
    return audio_path


def format_timestamp(seconds: float, sep: str = ",") -> str:
    td = timedelta(seconds=seconds)
    total = int(td.total_seconds())
    h, r = divmod(total, 3600)
    m, s = divmod(r, 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def push_status(task: dict):
    task["queue"].put(push_status_dict(task))


def format_transcription_output(segments_data: list[dict], result_style: str,
                                show_speakers: bool) -> str:
    lines = []
    for seg in segments_data:
        parts = []
        if result_style == "timestamp":
            parts.append(f"[{format_timestamp(seg['start'], sep='.')}]")
        if show_speakers and seg.get("speaker") is not None:
            parts.append(f"说话人{seg['speaker']}:")
        parts.append(seg["text"])
        lines.append(" ".join(parts))
    return "\n".join(lines)


def run_transcribe(task_id: str, file_path: str, model_name: str,
                   output_format: str, output_dir: str, output_name: str,
                   is_video: bool, speaker_count: str, result_style: str):
    task = tasks[task_id]
    temp_dir = None
    audio_to_process = file_path

    try:
        if is_video:
            task["status"] = STATUS_EXTRACTING
            task["message"] = "正在从视频中提取音频..."
            push_status(task)
            temp_dir = tempfile.mkdtemp(dir=TEMP_ROOT)
            audio_to_process = extract_audio(file_path, temp_dir, task_id)

        task["status"] = STATUS_LOADING
        task["message"] = "正在加载模型..."
        task["percent"] = 5
        push_status(task)

        from faster_whisper import WhisperModel

        model = WhisperModel(
            model_name,
            device="cpu",
            compute_type="int8",
            cpu_threads=os.cpu_count() or 4,
            download_root=str(MODELS_DIR),
        )

        task["status"] = STATUS_TRANSCRIBING
        task["message"] = "正在转写..."
        task["percent"] = 10
        push_status(task)

        segments_result, info = model.transcribe(
            audio_to_process,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
            ),
        )

        total_duration = info.duration
        segments = list(segments_result)

        last_pushed_pct = 10
        for seg in segments:
            if total_duration > 0:
                pct = min(int(seg.end / total_duration * 90) + 10, 99)
                if pct > last_pushed_pct:
                    task["percent"] = pct
                    task["message"] = f"转写中... {pct - 10}%"
                    last_pushed_pct = pct
                    push_status(task)

        # Speaker diarization
        do_diarize = speaker_count != "none"
        segments_data = []
        n_speakers_found = 0

        if do_diarize:
            task["status"] = STATUS_DIARIZING
            task["message"] = "正在识别说话人..."
            task["percent"] = 95
            push_status(task)

            diarize_wav = audio_to_process
            if not diarize_wav.lower().endswith(".wav"):
                diarize_temp_dir = tempfile.mkdtemp(dir=TEMP_ROOT)
                if temp_dir:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                temp_dir = diarize_temp_dir
                diarize_wav = extract_audio(file_path, temp_dir, task_id)

            from speaker_diarization import perform_diarization
            n_speakers_arg = None if speaker_count == "auto" else int(speaker_count)
            segments_data, n_speakers_found = perform_diarization(
                diarize_wav, segments, n_speakers=n_speakers_arg,
            )
        else:
            segments_data = [
                {"start": s.start, "end": s.end, "text": s.text.strip(), "speaker": None}
                for s in segments
            ]

        show_speakers = do_diarize and n_speakers_found > 0

        os.makedirs(output_dir, exist_ok=True)

        if output_format == "srt":
            srt_lines = []
            for i, seg in enumerate(segments_data, 1):
                start_ts = format_timestamp(seg["start"])
                end_ts = format_timestamp(seg["end"])
                srt_text = seg["text"]
                if show_speakers and seg.get("speaker") is not None:
                    srt_text = f"说话人{seg['speaker']}: {srt_text}"
                srt_lines.append(f"{i}\n{start_ts} --> {end_ts}\n{srt_text}\n")
            full_text = "\n".join(srt_lines)
            output_path = os.path.join(output_dir, f"{output_name}.srt")
        else:
            full_text = format_transcription_output(segments_data, result_style, show_speakers)
            output_path = os.path.join(output_dir, f"{output_name}.txt")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        task["status"] = STATUS_DONE
        task["percent"] = 100
        task["message"] = "转写完成！"
        task["output_path"] = output_path
        task["segment_count"] = len(segments)
        task["duration_seconds"] = round(total_duration, 1)
        task["detected_language"] = info.language
        task["preview_text"] = full_text[:5000]
        task["n_speakers"] = n_speakers_found if show_speakers else 0
        push_status(task)

    except Exception as e:
        task["status"] = STATUS_ERROR
        task["message"] = str(e)
        push_status(task)
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/transcribe", methods=["POST"])
def api_transcribe():
    data = request.form
    uploaded_file = request.files.get("file")

    model_name = data.get("model", "medium")
    if model_name not in MODELS:
        return jsonify({"error": f"不支持的模型: {model_name}"}), 400

    output_format = data.get("format", "txt")
    if output_format not in ("txt", "srt"):
        return jsonify({"error": f"不支持的输出格式: {output_format}"}), 400

    file_type = data.get("file_type", "audio")
    if file_type not in ("audio", "video"):
        return jsonify({"error": f"不支持的文件类型: {file_type}"}), 400

    speaker_count = data.get("speaker_count", "none")
    if speaker_count not in ("none", "auto") and not (
        speaker_count.isdigit() and 1 <= int(speaker_count) <= 9
    ):
        return jsonify({"error": "无效的说话人分离设置"}), 400

    result_style = data.get("result_style", "timestamp")
    if result_style not in ("timestamp", "no_timestamp"):
        return jsonify({"error": "无效的结果样式设置"}), 400

    if uploaded_file and uploaded_file.filename:
        task_id = str(uuid.uuid4())[:8]
        ext = Path(uploaded_file.filename).suffix
        save_dir = tempfile.mkdtemp(dir=TEMP_ROOT)
        file_path = os.path.join(save_dir, f"{task_id}_input{ext}")
        uploaded_file.save(file_path)
        input_name = Path(uploaded_file.filename).stem
        output_dir = data.get("output_dir", "").strip() or str(BASE_DIR)
        is_temp = True
    else:
        file_path = data.get("file_path", "").strip()
        if not file_path:
            return jsonify({"error": "请提供音频文件路径或上传文件"}), 400
        if not os.path.exists(file_path):
            return jsonify({"error": f"文件不存在: {file_path}"}), 400
        task_id = str(uuid.uuid4())[:8]
        input_name = Path(file_path).stem
        output_dir = data.get("output_dir", "").strip() or str(Path(file_path).parent)
        is_temp = False

    output_name = data.get("output_name", "").strip() or f"{input_name}录音转文字"
    is_video = file_type == "video"

    task = {
        "id": task_id,
        "status": STATUS_QUEUED,
        "message": "排队中...",
        "percent": 0,
        "output_path": None,
        "segment_count": 0,
        "duration_seconds": 0,
        "detected_language": None,
        "preview_text": "",
        "queue": queue.Queue(),
        "is_temp": is_temp,
        "temp_dir": os.path.dirname(file_path) if is_temp else None,
    }

    _cleanup_old_tasks()
    with tasks_lock:
        tasks[task_id] = task

    thread = threading.Thread(
        target=run_transcribe,
        args=(task_id, file_path, model_name, output_format, output_dir, output_name,
              is_video, speaker_count, result_style),
        daemon=True,
    )
    thread.start()

    return jsonify({"task_id": task_id})


def _sse_message(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _is_terminal(task: dict) -> bool:
    return task["status"] in (STATUS_DONE, STATUS_ERROR)


def _cleanup_old_tasks():
    stale = [tid for tid, t in tasks.items() if _is_terminal(t)]
    for tid in stale:
        del tasks[tid]


@app.route("/api/progress/<task_id>")
def api_progress(task_id):
    task = tasks.get(task_id)
    if not task:
        return jsonify({"error": "任务不存在"}), 404

    def generate():
        q = task["queue"]
        last_status = task["status"]
        while True:
            try:
                msg = q.get(timeout=1)
                yield _sse_message(msg)
                last_status = msg.get("status", last_status)
                if msg.get("status") in (STATUS_DONE, STATUS_ERROR):
                    break
            except queue.Empty:
                if task["status"] != last_status:
                    yield _sse_message(push_status_dict(task))
                    last_status = task["status"]
                if _is_terminal(task):
                    break

        if task.get("is_temp") and task.get("temp_dir"):
            shutil.rmtree(task["temp_dir"], ignore_errors=True)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def push_status_dict(task: dict) -> dict:
    return {
        "status": task["status"],
        "message": task["message"],
        "percent": task.get("percent", 0),
        "output_path": task.get("output_path"),
        "segment_count": task.get("segment_count", 0),
        "duration_seconds": task.get("duration_seconds", 0),
        "detected_language": task.get("detected_language"),
        "preview_text": task.get("preview_text", ""),
        "n_speakers": task.get("n_speakers", 0),
    }


@app.route("/api/read")
def api_read():
    path = request.args.get("path", "")
    if not path or not os.path.exists(path):
        return jsonify({"error": "文件不存在"}), 404
    try:
        return send_file(path, mimetype="text/plain; charset=utf-8")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health")
def api_health():
    global _MODEL_STATUS_CACHE
    if _MODEL_STATUS_CACHE is None:
        _MODEL_STATUS_CACHE = {}
        for m in MODELS:
            p = MODELS_DIR / f"models--Systran--faster-whisper-{m}"
            _MODEL_STATUS_CACHE[m] = p.exists() and any(p.rglob("*.bin"))
    return jsonify({
        "status": "ok",
        "ffmpeg_available": check_ffmpeg(),
        "models": list(MODELS),
        "models_cached": _MODEL_STATUS_CACHE,
        "models_dir": str(MODELS_DIR),
    })


if __name__ == "__main__":
    print("=" * 50)
    print("  离线语音转文字工具")
    print("  打开浏览器访问 http://127.0.0.1:5100")
    print("  模型存放目录:", MODELS_DIR)
    print("=" * 50)

    if not check_ffmpeg():
        print("[警告] 未检测到 ffmpeg，视频音频提取功能将不可用")
        print("  安装: winget install Gyan.FFmpeg")

    app.run(host="127.0.0.1", port=5100, debug=False, threaded=True)
