let currentFileType = 'audio';
let selectedFile = null;
let isRunning = false;

const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const pathInput = document.getElementById('pathInput');
const startBtn = document.getElementById('startBtn');
const progressSection = document.getElementById('progressSection');
const resultSection = document.getElementById('resultSection');
const toast = document.getElementById('toast');
const formatHint = document.getElementById('formatHint');
const themeToggle = document.getElementById('themeToggle');

const progressStatus = document.getElementById('progressStatus');
const progressPercent = document.getElementById('progressPercent');
const progressFill = document.getElementById('progressFill');
const progressMessage = document.getElementById('progressMessage');

const STATUS_LABELS = {
  extracting_audio: '提取音频',
  loading_model: '加载模型',
  transcribing: '转写中',
  diarizing: '说话人分离',
  queued: '排队中',
};

let currentEventSource = null;
let toastTimer = null;

const audioFormats = 'MP3 / M4A / WAV / FLAC / OGG / AAC';
const videoFormats = 'MP4 / MOV / AVI / MKV / WEBM';

// Theme

const THEME_KEY = 'whisper-theme';

function getTheme() {
  const saved = localStorage.getItem(THEME_KEY);
  return saved || 'light';
}

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  const icon = themeToggle.querySelector('.theme-icon');
  icon.textContent = theme === 'dark' ? '☀️' : '🌙';
  themeToggle.title = theme === 'dark' ? '切换到浅色主题' : '切换到深色主题';
  localStorage.setItem(THEME_KEY, theme);
}

themeToggle.addEventListener('click', () => {
  const next = getTheme() === 'dark' ? 'light' : 'dark';
  applyTheme(next);
});

applyTheme(getTheme());

// File type toggle

document.querySelectorAll('.toggle-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentFileType = btn.dataset.type;
    formatHint.textContent = '支持 ' + (currentFileType === 'video' ? videoFormats : audioFormats);
    updateStartButton();
  });
});

// Radio cards

const speakerStyleGroup = document.getElementById('speakerStyleGroup');

function updateSpeakerStyleVisibility() {
  const isSrt = getSelectedRadioValue('format', 'txt') === 'srt';
  speakerStyleGroup.style.display = isSrt ? 'none' : '';
}
updateSpeakerStyleVisibility();

document.querySelectorAll('.radio-card').forEach(card => {
  card.addEventListener('click', () => {
    const name = card.querySelector('input').name;
    document.querySelectorAll(`input[name="${name}"]`).forEach(inp => {
      inp.closest('.radio-card').classList.remove('active');
    });
    card.classList.add('active');
    card.querySelector('input').checked = true;
    if (name === 'format') updateSpeakerStyleVisibility();
  });
});

// Drag and drop

dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  if (e.dataTransfer.files.length > 0) {
    handleFile(e.dataTransfer.files[0]);
  }
});

fileInput.addEventListener('change', () => {
  if (fileInput.files.length > 0) {
    handleFile(fileInput.files[0]);
  }
});

function handleFile(file) {
  selectedFile = file;
  const el = dropZone.querySelector('.drop-title');
  el.textContent = file.name;
  dropZone.querySelector('.drop-subtitle').textContent =
    (file.size / 1024 / 1024).toFixed(1) + ' MB';
  dropZone.classList.add('has-file');
  pathInput.value = '';
  updateStartButton();
}

// Path input

let pathDebounce = null;
pathInput.addEventListener('input', () => {
  if (pathInput.value.trim()) {
    selectedFile = null;
    dropZone.querySelector('.drop-title').textContent = '拖拽文件到此处';
    dropZone.querySelector('.drop-subtitle').textContent = '或点击选择文件';
    dropZone.classList.remove('has-file');
    fileInput.value = '';
  }
  clearTimeout(pathDebounce);
  pathDebounce = setTimeout(updateStartButton, 200);
});

// Start button state

function updateStartButton() {
  startBtn.disabled = !(selectedFile || pathInput.value.trim());
}

// Start transcription

startBtn.addEventListener('click', startTranscription);

function getSelectedRadioValue(name, defaultVal) {
  const checked = document.querySelector(`input[name="${name}"]:checked`);
  return checked ? checked.value : defaultVal;
}

async function startTranscription() {
  if (isRunning) return;
  isRunning = true;
  startBtn.disabled = true;
  startBtn.querySelector('span').textContent = '转写中...';

  hideResult();
  showProgress();
  updateProgress(0, '启动中...', 'processing');

  const formData = new FormData();
  formData.append('model', getSelectedRadioValue('model', 'medium'));
  formData.append('format', getSelectedRadioValue('format', 'txt'));
  formData.append('file_type', currentFileType);
  formData.append('speaker_count', document.getElementById('speakerCount').value);
  formData.append('result_style', document.getElementById('resultStyle').value);

  if (selectedFile) {
    formData.append('file', selectedFile);
  } else {
    formData.append('file_path', pathInput.value.trim());
  }

  const outputDir = document.getElementById('outputDir').value.trim();
  const outputName = document.getElementById('outputName').value.trim();
  if (outputDir) formData.append('output_dir', outputDir);
  if (outputName) formData.append('output_name', outputName);

  try {
    const resp = await fetch('/api/transcribe', { method: 'POST', body: formData });

    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(text || '服务器返回错误 ' + resp.status);
    }

    const data = await resp.json();

    if (data.error) {
      showError(data.error);
      return;
    }

    listenProgress(data.task_id);
  } catch (err) {
    showError(err.message);
  }
}

function listenProgress(taskId) {
  if (currentEventSource) {
    currentEventSource.close();
  }
  currentEventSource = new EventSource(`/api/progress/${taskId}`);

  currentEventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);

      if (data.status === 'done') {
        updateProgress(100, '转写完成！', 'done');
        currentEventSource.close();
        showResult(data);
        resetAfterDone();
      } else if (data.status === 'error') {
        currentEventSource.close();
        showError(data.message || '转写失败');
      } else {
        updateProgress(data.percent || 0, data.message, data.status);
      }
    } catch (e) {
      currentEventSource.close();
      showError('解析进度数据失败');
    }
  };

  currentEventSource.onerror = () => {
    currentEventSource.close();
    resetAfterDone();
  };
}

// Progress UI

function showProgress() {
  progressSection.hidden = false;
}

function updateProgress(pct, message, status) {
  progressStatus.textContent = STATUS_LABELS[status] || '处理中';
  progressPercent.textContent = pct + '%';
  progressFill.style.width = pct + '%';
  progressMessage.textContent = message;
}

// Result UI

function showResult(data) {
  document.getElementById('resultSegments').textContent = data.segment_count;
  document.getElementById('resultDuration').textContent = data.duration_seconds;
  document.getElementById('resultLanguage').textContent = data.detected_language || '--';
  document.getElementById('resultPath').textContent = data.output_path || '--';
  document.getElementById('resultPreview').textContent = data.preview_text || '';

  resultSection.hidden = false;
  resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function hideResult() {
  resultSection.hidden = true;
}

document.getElementById('closeResult').addEventListener('click', () => {
  resultSection.hidden = true;
});

// Error handling

function showError(msg) {
  progressSection.hidden = true;
  toast.textContent = msg;
  toast.hidden = false;
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toast.hidden = true; }, 5000);
  resetAfterDone();
}

function resetAfterDone() {
  isRunning = false;
  startBtn.disabled = false;
  startBtn.querySelector('span').textContent = '开始转写';
  updateStartButton();
}

// Health check on load

fetch('/api/health')
  .then(r => r.json())
  .then(data => {
    const statusEl = document.getElementById('systemStatus');
    if (!data.ffmpeg_available) {
      statusEl.querySelector('.status-dot').style.background = '#f0a030';
      statusEl.querySelector('.status-dot').style.boxShadow = '0 0 6px #f0a040';
      statusEl.querySelector('.status-text').textContent = 'ffmpeg 未安装';
      statusEl.title = '视频音频提取不可用，请安装 ffmpeg';
    }
  })
  .catch(() => {});
