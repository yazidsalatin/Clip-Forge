# Script PowerShell untuk menjalankan server ClipForge
Set-Location -Path "$PSScriptRoot\server"

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "      ClipForge - Web Siap Pakai (Mode Mandiri)        " -ForegroundColor Green
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host ""

Start-Process "http://127.0.0.1:8000"
Write-Host "Server berjalan di http://127.0.0.1:8000 (atau http://localhost:8000) ..." -ForegroundColor Yellow
python app.py
