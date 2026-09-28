@echo off
rem Double-click this on Windows to start Haven and open its window. The first time, it sets itself up:
rem that needs the internet and takes a few minutes. It uses uv (https://docs.astral.sh/uv/) to get Python
rem and Haven's parts, which go into the .venv folder here, so nothing else on this computer is changed.
setlocal
cd /d "%~dp0"
title Haven

set "UV=%USERPROFILE%\.local\bin\uv.exe"
if exist "%UV%" goto have_uv
where uv >nul 2>nul && set "UV=uv" && goto have_uv
echo Getting uv, a small tool that installs Python and Haven's parts...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$env:UV_INSTALL_DIR = Join-Path $env:USERPROFILE '.local\bin'; $env:UV_NO_MODIFY_PATH = '1'; irm https://astral.sh/uv/install.ps1 | iex"
if exist "%UV%" goto have_uv
echo Couldn't get uv. Check the internet connection, or install uv yourself (https://docs.astral.sh/uv/), then try again.
goto failed

:have_uv
if exist ".venv\Scripts\haven.exe" goto run
echo Setting Haven up. The first time takes a few minutes...
if exist ".venv\Scripts\python.exe" goto install
"%UV%" venv --python 3.12 .venv || goto failed
:install
rem --torch-backend=auto picks the PyTorch build for this computer's graphics card, if it has one.
"%UV%" pip install --python .venv\Scripts\python.exe --torch-backend=auto -e ".[cortex]" || "%UV%" pip install --python .venv\Scripts\python.exe -e ".[cortex]" || goto failed

:run
".venv\Scripts\haven.exe" app %*
if errorlevel 1 pause
exit /b

:failed
echo.
echo Haven couldn't be set up; the messages above say why. Check the internet connection and try again.
pause
exit /b 1
