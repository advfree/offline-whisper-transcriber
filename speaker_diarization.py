"""Offline speaker diarization using pure NumPy (no scipy/sklearn/librosa needed).

Given a 16kHz mono WAV and a list of transcription segments from faster-whisper,
assigns a speaker label to each segment via MFCC feature extraction + K-means clustering.
"""

import wave
import numpy as np

_SAMPLE_RATE = 16000
_FRAME_SIZE = 0.025
_FRAME_SHIFT = 0.010
_N_FFT = 512
_N_MELS = 26
_N_MFCC = 13
_KMEANS_SUBSAMPLE_THRESH = 30000
_ELBOW_THRESHOLD = 0.15

_DCT_BASIS: np.ndarray | None = None
_MEL_FB: np.ndarray | None = None


def _get_dct_basis() -> np.ndarray:
    global _DCT_BASIS
    if _DCT_BASIS is None:
        basis = np.zeros((_N_MFCC, _N_MELS), dtype=np.float32)
        for k in range(_N_MFCC):
            basis[k] = np.cos(np.pi * k * (np.arange(_N_MELS) + 0.5) / _N_MELS)
        basis[0] *= np.sqrt(1.0 / _N_MELS)
        basis[1:] *= np.sqrt(2.0 / _N_MELS)
        _DCT_BASIS = basis
    return _DCT_BASIS


def _get_mel_fb(n_fft: int = _N_FFT, sample_rate: int = _SAMPLE_RATE,
                n_mels: int = _N_MELS) -> np.ndarray:
    global _MEL_FB
    if _MEL_FB is not None:
        return _MEL_FB
    fmax = sample_rate / 2.0
    n_bins = n_fft // 2 + 1
    mel_low = _hz_to_mel(np.array(0.0, dtype=np.float32))
    mel_high = _hz_to_mel(np.array(fmax, dtype=np.float32))
    mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bin_idx = np.floor((n_fft + 1) * hz_points / sample_rate).astype(np.int32)
    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for m in range(1, n_mels + 1):
        lo, mid, hi = bin_idx[m - 1], bin_idx[m], bin_idx[m + 1]
        if mid > lo:
            fb[m - 1, lo:mid] = (np.arange(lo, mid) - lo) / (mid - lo)
        if hi > mid:
            fb[m - 1, mid:hi] = (hi - np.arange(mid, hi)) / (hi - mid)
    _MEL_FB = fb
    return fb


def read_wav(wav_path: str) -> np.ndarray:
    """Read a PCM WAV file, return float32 samples in [-1, 1]."""
    with wave.open(wav_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        dtype = np.int16
    elif sampwidth == 1:
        dtype = np.uint8
    elif sampwidth == 4:
        dtype = np.int32
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)
    max_val = float(np.iinfo(dtype).max) if dtype != np.uint8 else 128.0
    audio /= max_val
    return audio


def _hamming_window(n: int) -> np.ndarray:
    i = np.arange(n, dtype=np.float32)
    return 0.54 - 0.46 * np.cos(2.0 * np.pi * i / (n - 1))


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def compute_mfcc(audio: np.ndarray, sample_rate: int = _SAMPLE_RATE,
                 n_mfcc: int = _N_MFCC, n_mels: int = _N_MELS,
                 frame_size: float = _FRAME_SIZE, frame_shift: float = _FRAME_SHIFT,
                 n_fft: int = _N_FFT) -> np.ndarray:
    if audio.size == 0:
        return np.empty((0, n_mfcc * 2), dtype=np.float32)

    emphasised = np.copy(audio)
    emphasised[1:] -= 0.97 * audio[:-1]
    emphasised[0] *= 0.03

    frame_len = int(sample_rate * frame_size)
    frame_step = int(sample_rate * frame_shift)

    if len(emphasised) < frame_len:
        pad = np.zeros(frame_len - len(emphasised), dtype=np.float32)
        emphasised = np.concatenate([emphasised, pad])

    try:
        frames = np.lib.stride_tricks.sliding_window_view(emphasised, frame_len)[::frame_step]
    except Exception:
        n_frames = max(1, (len(emphasised) - frame_len) // frame_step + 1)
        idx = np.arange(frame_len)[None, :] + np.arange(n_frames)[:, None] * frame_step
        frames = emphasised[idx]

    window = _hamming_window(frame_len)
    frames = frames * window.astype(np.float32)

    mag = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)).astype(np.float32)
    np.square(mag, out=mag)
    power = mag

    mel_fb = _get_mel_fb(n_fft, sample_rate, n_mels)
    mel_energy = mel_fb @ power.T

    mel_energy = np.log(np.maximum(mel_energy, 1e-10))

    mfcc = _get_dct_basis() @ mel_energy

    mfcc_t = mfcc.T
    delta = np.zeros_like(mfcc_t)
    delta[2:-2] = (mfcc_t[4:] - mfcc_t[:-4]) / 2.0
    delta[1] = mfcc_t[2] - mfcc_t[0]
    delta[-2] = mfcc_t[-1] - mfcc_t[-3]
    delta[0] = delta[1]
    delta[-1] = delta[-2]

    return np.concatenate([mfcc_t, delta], axis=1).astype(np.float32)


def _kmeans_plusplus_init(X: np.ndarray, n_clusters: int, rng: np.random.Generator) -> np.ndarray:
    """K-means++ centroid initialization."""
    n_samples = X.shape[0]
    centroids = np.zeros((n_clusters, X.shape[1]), dtype=X.dtype)

    first = rng.integers(n_samples)
    centroids[0] = X[first]

    for k in range(1, n_clusters):
        dist_sq = np.sum((X[:, None, :] - centroids[None, :k, :]) ** 2, axis=2)
        min_dist = np.min(dist_sq, axis=1)
        probs = min_dist / np.sum(min_dist)
        cumprobs = np.cumsum(probs)
        r = rng.random()
        next_idx = int(np.searchsorted(cumprobs, r))
        centroids[k] = X[next_idx]

    return centroids


def kmeans(X: np.ndarray, n_clusters: int, max_iters: int = 100,
           tol: float = 1e-4, n_init: int = 10,
           random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Pure-numpy K-means clustering. Returns (labels, centroids)."""
    if n_clusters >= X.shape[0]:
        labels = np.arange(X.shape[0]) % n_clusters
        centroids = np.array([X[labels == k].mean(axis=0) if np.any(labels == k) else X.mean(axis=0)
                             for k in range(n_clusters)])
        return labels.astype(np.int32), centroids

    rng = np.random.default_rng(random_state)
    best_inertia = np.inf
    best_labels = None
    best_centroids = None
    x_norm = np.sum(X ** 2, axis=1, keepdims=True)

    for _ in range(n_init):
        centroids = _kmeans_plusplus_init(X, n_clusters, rng)
        for _ in range(max_iters):
            c_norm = np.sum(centroids ** 2, axis=1, keepdims=True)
            dist_sq = x_norm + c_norm.T - 2.0 * (X @ centroids.T)
            dist_sq = np.maximum(dist_sq, 0.0)
            labels = np.argmin(dist_sq, axis=1).astype(np.int32)

            new_centroids = np.zeros_like(centroids)
            for k in range(n_clusters):
                mask = labels == k
                if mask.any():
                    new_centroids[k] = X[mask].mean(axis=0)
                else:
                    far_idx = np.argmax(np.min(dist_sq, axis=1))
                    new_centroids[k] = X[far_idx]
                    labels[far_idx] = k

            shift = np.sum((new_centroids - centroids) ** 2)
            centroids = new_centroids
            if shift < tol:
                break

        inertia = np.sum((X - centroids[labels]) ** 2)
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centroids = centroids.copy()

    return best_labels, best_centroids


def auto_determine_speakers(features: np.ndarray, max_speakers: int = 9) -> int:
    """Estimate number of speakers via elbow method on K-means distortion curve."""
    if features.shape[0] < 2:
        return 1
    max_k = min(max_speakers, features.shape[0])
    if max_k <= 1:
        return 1

    distortions = []
    for k in range(1, max_k + 1):
        if k == 1:
            mean = features.mean(axis=0)
            distortions.append(float(np.sum((features - mean) ** 2)))
        else:
            labels, centroids = kmeans(features, k, n_init=3, random_state=42)
            distortions.append(float(np.sum((features - centroids[labels]) ** 2)))

    delta = [distortions[i] - distortions[i + 1] for i in range(len(distortions) - 1)]
    if not delta or delta[0] <= 0:
        return 1

    threshold = delta[0] * _ELBOW_THRESHOLD
    for k, d in enumerate(delta):
        if d < threshold:
            return max(1, k + 1)

    return max(1, max_k - 1)


def _seg_to_dict(seg, speaker: int | None = None) -> dict:
    if isinstance(seg, dict):
        return {**seg, "speaker": speaker}
    return {
        "start": seg.start,
        "end": seg.end,
        "text": seg.text.strip(),
        "speaker": speaker,
    }


def perform_diarization(wav_path: str, segments: list,
                        n_speakers: int | None = None,
                        sample_rate: int = _SAMPLE_RATE) -> tuple[list[dict], int]:
    """Main diarization entry point.

    Args:
        wav_path: Path to 16kHz mono WAV file.
        segments: List of dicts with keys 'start', 'end', 'text'.
        n_speakers: None for auto-detect, 0 for no diarization, or a positive int.

    Returns:
        (result_segments, actual_speaker_count) where result_segments is a list of
        dicts with keys 'start', 'end', 'text', 'speaker'.
    """
    _no_dia = ([_seg_to_dict(s) for s in segments], 0)

    if n_speakers == 0 or not segments:
        return _no_dia

    audio = read_wav(wav_path)
    duration = len(audio) / sample_rate
    if duration < 0.5:
        return _no_dia

    features = compute_mfcc(audio, sample_rate)
    if features.shape[0] < 2:
        return _no_dia

    # Normalise features per dimension
    feat_mean = features.mean(axis=0)
    feat_std = features.std(axis=0)
    feat_std[feat_std == 0] = 1.0
    features_norm = (features - feat_mean) / feat_std

    if n_speakers is None:
        n_speakers = auto_determine_speakers(features_norm)
    if n_speakers <= 1:
        return _no_dia

    n_frames = features_norm.shape[0]
    subsample = max(1, n_frames // _KMEANS_SUBSAMPLE_THRESH)
    x_norm_feat = np.sum(features_norm ** 2, axis=1, keepdims=True)

    if subsample > 1:
        features_sub = features_norm[::subsample]
    else:
        features_sub = features_norm

    labels_all, _ = kmeans(features_sub, n_speakers)

    if subsample > 1:
        centroids = np.array([features_sub[labels_all == k].mean(axis=0)
                              for k in range(n_speakers)])
        c_norm = np.sum(centroids ** 2, axis=1, keepdims=True)
        dist_sq = x_norm_feat + c_norm.T - 2.0 * (features_norm @ centroids.T)
        labels_all = np.argmin(np.maximum(dist_sq, 0.0), axis=1).astype(np.int32)

    # Normalize segments to dict and assign speakers
    segs = [_seg_to_dict(s) for s in segments]
    frames_per_shift = sample_rate * _FRAME_SHIFT
    result = []
    for seg in segs:
        start_frame = int(seg["start"] * sample_rate / frames_per_shift)
        end_frame = int(seg["end"] * sample_rate / frames_per_shift) + 1
        start_frame = max(0, min(start_frame, len(labels_all) - 1))
        end_frame = max(start_frame + 1, min(end_frame, len(labels_all)))

        seg_labels = labels_all[start_frame:end_frame]
        speaker = int(np.argmax(np.bincount(seg_labels))) + 1 if len(seg_labels) else 1
        seg["speaker"] = speaker
        result.append(seg)

    return result, n_speakers
