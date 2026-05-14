# 离线语音转文字工具

基于 [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2) 的 Windows 离线语音识别 Web 应用。纯本地运行，无需联网，数据安全。

## 功能

- 支持**音频**（MP3/M4A/WAV/FLAC/OGG/AAC）和**视频**（MP4/MOV/AVI/MKV/WEBM）
- 两种模型可选：Medium（均衡推荐）/ Large-V3（最高精度）
- 输出格式：TXT（纯文本）/ SRT（字幕格式）
- **说话人分离**：不分离 / 自动检测 / 指定 1-9 人（纯 NumPy 实现，无需额外依赖）
- **时间戳控制**：含时间戳 / 不含时间戳
- 浅色/深色主题
- SSE 实时进度推送

## 系统要求

- Windows 10+
- Python 3.10+
- [ffmpeg](https://ffmpeg.org/)（视频音频提取必需，可选）

### 安装 ffmpeg

```
winget install Gyan.FFmpeg
```

## 安装

```bash
pip install -r requirements.txt
```

## 模型下载

首次运行时，faster-whisper 会自动从 Hugging Face 下载模型到 `models/` 目录。你也可以手动下载：

| 模型 | 大小 | 下载地址 |
|------|------|----------|
| Medium（推荐） | ~1.5 GB | [Systran/faster-whisper-medium](https://huggingface.co/Systran/faster-whisper-medium) |
| Large-V3（高精度） | ~3 GB | [Systran/faster-whisper-large-v3](https://huggingface.co/Systran/faster-whisper-large-v3) |

手动下载步骤：

1. 访问上述 Hugging Face 链接，下载所有文件到 `models/models--Systran--faster-whisper-<模型名>/snapshots/<hash>/` 目录
2. 或者直接运行一次转写任务，让程序自动下载

## 启动

```bash
python app.py
```

浏览器访问 **http://127.0.0.1:5100**

或双击 `启动.bat` 自动打开浏览器。

## 技术栈

- **后端**: Flask + faster-whisper (CTranslate2)
- **前端**: 原生 HTML/CSS/JS，SSE 实时推送
- **说话人分离**: 纯 NumPy 实现（MFCC 特征提取 + K-means++ 聚类 + 肘部法则自动判断说话人数）
- **模型量化**: CPU / int8，自动匹配线程数

## 许可证

MIT
