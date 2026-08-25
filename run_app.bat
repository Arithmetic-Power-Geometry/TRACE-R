@echo off
python -m pip install -r requirements.txt
set PYTHONPATH=src
streamlit run app.py
