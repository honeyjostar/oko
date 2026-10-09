@echo off
chcp 65001 >nul
cd /d "%~dp0\.."
echo === OKO: установка ===
set PY=python
where py >nul 2>nul && set PY=py -3
%PY% --version >nul 2>nul || (
  echo Python не найден. Поставь Python 3.11 или новее с python.org и отметь "Add python.exe to PATH".
  if /i not "%~1"=="nopause" pause
  exit /b 1
)
if not exist .venv (
  echo Создаю виртуальное окружение .venv
  %PY% -m venv .venv || (pause & exit /b 1)
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip >nul
pip install -r requirements.txt || (pause & exit /b 1)
echo.
python -m oko doctor
echo.
echo Если Tesseract не найден: поставь его с github.com/UB-Mannheim/tesseract/wiki
echo и в установщике отметь русский язык (Additional language data - Russian).
echo.
echo Готово. Демо: scripts\demo.bat
if /i not "%~1"=="nopause" pause
