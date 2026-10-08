#!/usr/bin/env bash
cd "$(dirname "$0")/.."
[ -d .venv ] || scripts/setup.sh
. .venv/bin/activate
exec streamlit run app.py
