#!/usr/bin/env bash
set -euo pipefail
python -m pip install -r requirements.txt
PYTHONPATH=src streamlit run app.py
