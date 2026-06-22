@echo off
title Detector YOLO11n

echo ============================================
echo   Detector de Objetos YOLO11n - Local
echo ============================================
echo.

REM Verificar Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python no encontrado.
    echo Descargalo desde https://www.python.org/downloads/
    echo Asegurate de marcar "Add Python to PATH" al instalar.
    pause
    exit /b
)

REM Crear entorno virtual si no existe
if not exist "venv\" (
    echo [1/3] Creando entorno virtual...
    python -m venv venv
)

REM Activar entorno
call venv\Scripts\activate.bat

REM Instalar dependencias
echo [2/3] Instalando dependencias ^(solo la primera vez^)...
pip install -r requirements.txt -q

REM Abrir navegador automaticamente
echo [3/3] Iniciando servidor...
echo.
echo  Abre tu navegador en: http://localhost:5000
echo  Presiona Ctrl+C para detener.
echo.
start "" "http://localhost:5000"

python app.py

pause
