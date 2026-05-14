@echo off
chcp 65001 >nul
echo 正在启动离线语音转文字服务...
echo.
start http://127.0.0.1:5100
python app.py
pause
