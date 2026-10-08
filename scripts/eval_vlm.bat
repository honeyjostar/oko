@echo off
chcp 65001 >nul
cd /d "%~dp0\.."
call .venv\Scripts\activate.bat
echo === OKO: замер качества модели ===
echo Сервер модели должен быть запущен: LM Studio - Developer - Start Server (порт 1234)
echo.
python -m oko doctor
if not exist data\labels.jsonl (
  echo Создаю набор данных: 300 машин, 900 документов. Это минут 10-15.
  python -m oko generate --out data --vehicles 300 || (pause & exit /b 1)
)
echo.
echo [1/3] Сканы и фото тестовой части
python -m oko eval --engine vlm --data data || (pause & exit /b 1)
echo.
echo [2/3] Стресс-тест: 12 искажений x 4 уровня
python -m oko robustness --engine vlm --data data --per-type 3
echo.
echo [3/3] Отчёт
python -m oko report --data data
echo.
echo Готово: docs\REPORT.md. Прогон можно прервать и запустить снова - готовое не пересчитывается.
pause
