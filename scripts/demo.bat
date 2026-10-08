@echo off
chcp 65001 >nul
cd /d "%~dp0\.."
if not exist .venv (call scripts\setup.bat)
call .venv\Scripts\activate.bat
streamlit run app.py
