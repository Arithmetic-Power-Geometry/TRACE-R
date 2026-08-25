.PHONY: test benchmark experiments all app external

test:
	PYTHONPATH=src pytest -q

benchmark:
	PYTHONPATH=src python scripts/run_benchmark.py

experiments:
	PYTHONPATH=src python scripts/run_extended_experiments.py
	PYTHONPATH=src python scripts/run_resilience_experiments.py

all: test benchmark experiments

external:
	PYTHONPATH=src python scripts/run_external_validation.py --who-when

app:
	streamlit run app.py
