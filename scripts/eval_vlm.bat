@echo off
chcp 65001 >nul
cd /d "%~dp0\.."
if not exist .venv\Scripts\activate.bat (
  echo Первый запуск: ставлю зависимости...
  call scripts\setup.bat nopause || (pause & exit /b 1)
)
call .venv\Scripts\activate.bat
echo === OKO: замер качества модели ===
echo Сервер модели должен быть запущен: LM Studio - Developer - Start Server (порт 1234)
echo.
python -m oko doctor
python -c "from oko.engines import get_engine; import sys; sys.exit(0 if get_engine('vlm').available()[0] else 1)" || (
  echo.
  echo Модель не запущена. Открой LM Studio, загрузи Qwen3-VL 8B и нажми Start Server на вкладке Developer.
  echo Потом запусти этот файл ещё раз.
  pause & exit /b 1
)
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
