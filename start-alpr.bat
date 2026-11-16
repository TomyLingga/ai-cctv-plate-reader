@echo off
rem Start the alpr-id service by double-clicking this file.
rem The window stays open while the service runs. Close it or press Ctrl+C to stop.
setlocal EnableExtensions
title alpr-id
cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
    echo [alpr-id] uv belum terpasang atau tidak ada di PATH.
    echo           Pasang dari https://docs.astral.sh/uv/ lalu jalankan file ini lagi.
    goto :fail
)
if not exist ".env" (
    echo [alpr-id] File .env belum ada.
    echo           Salin .env.example menjadi .env lalu isi URL kamera.
    goto :fail
)
if not exist "config\app.yaml" (
    echo [alpr-id] File config\app.yaml belum ada.
    echo           Salin config\app.example.yaml menjadi config\app.yaml.
    goto :fail
)
if not exist "models\detector\plat.onnx" (
    echo [alpr-id] Model detektor models\detector\plat.onnx belum ada. Lihat README bagian "Model".
    goto :fail
)

rem ALPR_EXTRA in .env selects the ONNX Runtime build: cpu or gpu.
set "ALPR_EXTRA=cpu"
for /f "usebackq tokens=1,* delims==" %%A in (`findstr /b /i "ALPR_EXTRA=" ".env"`) do set "ALPR_EXTRA=%%B"

echo [alpr-id] Menjalankan service dengan ONNX Runtime %ALPR_EXTRA%.
echo [alpr-id] Tutup jendela ini atau tekan Ctrl+C untuk berhenti. Log: data\logs\alpr.log
echo.
uv run --frozen --extra %ALPR_EXTRA% alpr-id serve
if errorlevel 1 goto :fail
exit /b 0

:fail
echo.
echo [alpr-id] Service tidak berjalan. Pesan di atas menjelaskan penyebabnya.
pause
exit /b 1
