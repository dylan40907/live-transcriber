@echo off
rem One-time setup: makes .venv, installs packages, adds the Desktop and Start menu shortcuts.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe py -3 -m venv .venv
.venv\Scripts\python -m pip install -q --upgrade pip
.venv\Scripts\python -m pip install -q -r requirements.txt
where nvidia-smi >nul 2>nul && .venv\Scripts\python -m pip install -q -r requirements-gpu.txt
.venv\Scripts\python make_shortcut.py
echo Done.
