# Responsibility by Construction / TRACE-R

Reproducible research software for **Responsibility as an Error-Correcting Information Problem: Optimal Responsibility Codes, Legal Observability, and Verifiable Evidence in Autonomous Digital Systems**.

TRACE-R treats responsibility-identifying evidence as an error-correcting information architecture. The package combines AutoResponsibilityBench controlled incidents, abstaining reconstruction, information-theoretic observability, Responsibility Capacity, Responsibility Distance, mixed error-erasure coding, minimum-cost code design, adaptive evidence attacks, Byzantine witness recovery, Ed25519 receipts, replay protection, Merkle checkpoints, DREP evidence-class separation, privacy/cost trade-offs, risk-coverage analysis, transferability stress tests, and false-attribution-under-non-identifiability evaluation.

## Github reproduction

Upload this folder to GitHub. Choose **Actions -> Reproduce Responsibility by Construction -> Run workflow**. The workflow runs tests and regenerates every controlled result and figure.

Local equivalent:

```bash
python -m pip install -r requirements.txt
make all
```

## Interactive app

```bash
streamlit run app.py
```

The app exposes incident family, random evidence loss, adaptive channel deletion, witness hardening, abstention threshold, repeated and optimized responsibility codes, erasure/corruption budgets, signed-receipt verification, replay rejection, DREP export, telemetry cost, Fano bounds, and packaged result tables.

## External validation workflow

A separate manual GitHub Action, **External WhoWhen validation**, downloads the official Who&When dataset on a connected runner and performs a schema/integrity interoperability audit. External corpora are not redistributed. The manifest also records AgentDojo, GRADE, MP-Bench, TraceElephant, AgenTracer, CHIEF, and Proof-or-Stop as comparison/validation targets.

The packaged manuscript does **not** claim external results that were not executed in the offline build environment. This separation is deliberate scientific hygiene.

## Main advanced components

- Mixed error-erasure guarantee: `2t + b < d_R^min`.
- Minimum-cost responsibility-code design solved as a binary set-multicover MILP.
- Adaptive worst-case evidence erasure.
- Byzantine witness recovery under authenticated independent domains.
- Empirical identifiability interval: Fano converse, exact discrete MAP error, and Bhattacharyya union upper bound.
- Responsibility Transferability stress test based on Jensen-Shannon divergence.
- False Attribution Rate under Non-Identifiability (FAR-NI).
- Selective-prediction risk-coverage curve and AURC.

## Scientific and legal boundary

TRACE-R reconstructs and evaluates preserved evidence. It does not manufacture missing evidence, determine legal liability, guarantee court admissibility, or treat cryptographic integrity as proof of semantic truth. AutoResponsibilityBench results validate controlled behavior, not court accuracy.


## Citation

Akhtar, M. A. K. (2026). Responsibility as an Error-Correcting Information Problem: Optimal Responsibility Codes, Legal Observability, and Verifiable Evidence in Autonomous Digital Systems. Zenodo. https://doi.org/10.5281/zenodo.22098853

## License

Copyright (C) 2026 Mohammad Amir Khusru Akhtar.

Licensed under the Apache License, Version 2.0. See `LICENSE` and `NOTICE`.

