@echo off
chcp 65001 >nul
cd /d "%~dp0\.."
call .venv\Scripts\activate.bat
if not exist data\labels.jsonl python -m oko generate --out data --vehicles 300
python -m oko eval --engine tesseract --data data
python -m oko robustness --engine tesseract --data data --per-type 6
python -m oko report --data data
pause
