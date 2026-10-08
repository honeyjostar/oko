#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
python3 -m venv .venv
. .venv/bin/activate
pip install -q --upgrade pip
pip install -r requirements.txt
python -m oko doctor || true
echo "Tesseract: sudo apt install tesseract-ocr tesseract-ocr-rus  (macOS: brew install tesseract tesseract-lang)"
echo "Демо: scripts/demo.sh"
