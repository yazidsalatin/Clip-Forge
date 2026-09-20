@echo off
title ClipForge - AI Video Clipper (Mode Mandiri)
cd /d "%~dp0\server"

echo ========================================================
echo       ClipForge - Menyiapkan Web Siap Pakai
echo          Mode: 100%% Mandiri (Tanpa API Key)
echo ========================================================
echo.

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python tidak ditemukan di sistem. Pastikan Python 3.10+ terpasang di PATH.
    pause
    exit /b 1
)

echo Membuka browser ke http://127.0.0.1:8000 ...
start "" http://127.0.0.1:8000

echo Menjalankan server ClipForge...
python app.py

pause
