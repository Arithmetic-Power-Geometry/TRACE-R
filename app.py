# Copyright (C) 2026 Mohammad Amir Khusru Akhtar
# Licensed under the Apache License, Version 2.0.

import json
import sys
from itertools import combinations
from pathlib import Path

# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

# Streamlit Cloud executes app.py from the repository root.
# TRACE-R uses a src/ layout, so add src/ to Python's import path.
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# ---------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------

try:
    import pandas as pd
    import streamlit as st
except ImportError as exc:
    raise SystemExit(
        "Required dependencies are missing. "
        "Install them with: pip install -r requirements.txt"
    ) from exc

# ---------------------------------------------------------------------
# TRACE-R imports
# ---------------------------------------------------------------------

from trace_r.simulator import (
    FAMILIES,
    generate_scenario,
    retain_evidence,
)

from trace_r.reconstruct import reconstruct
from trace_r.learned import train_evidence_model
from trace_r.core import verify_chain
from trace_r.evidence import build_drep

from trace_r.theory import (
    channel_cost,
    fano_error_lower_bound,
)

from trace_r.coding import (
    repetition_codebook,
    summarize_codebook,
    erasure_guarantee_holds,
    optimize_responsibility_code,
    mixed_error_erasure_guarantee,
)

from trace_r.crypto import (
    SigningIdentity,
    signed_cross_domain_receipt,
    verify_signed_record,
    ReplayGuard,
    merkle_root,
)

# ---------------------------------------------------------------------
# Streamlit configuration
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="TRACE-R Responsibility Coding Lab",
    page_icon="🔎",
    layout="wide",
)

st.title("TRACE-R: Responsibility Coding, Resilience & Legal Observability Lab")

st.caption(
    "Research demonstrator for responsibility-identifiable system design. "
    "TRACE-R reconstructs and tests preserved evidence; it does not decide "
    "legal liability, evidentiary admissibility, or judicial responsibility."
)

# ---------------------------------------------------------------------
# Cached learned model
# ---------------------------------------------------------------------

@st.cache_resource
def load_model():
    return train_evidence_model(
        seed=7301,
        n_per_family=55,
    )


M = load_model()

# ---------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------

with st.sidebar:
    st.header("Scenario Controls")

    family = st.selectbox(
        "Incident family",
        FAMILIES,
    )

    hardened = st.toggle(
        "Witness-hardened evidence architecture",
        value=False,
    )

    retention = st.slider(
        "Random evidence retention",
        min_value=0.10,
        max_value=1.00,
        value=0.90,
        step=0.05,
    )

    threshold = st.slider(
        "Abstention threshold",
        min_value=0.10,
        max_value=0.95,
        value=0.58,
        step=0.01,
    )

    seed = st.number_input(
        "Seed",
        min_value=0,
        max_value=1_000_000,
        value=42,
        step=1,
    )

    all_channels = [
        "authorization",
        "delegation",
        "permission",
        "action",
        "warning",
        "intervention",
        "policy",
        "identity",
        "model_version",
        "receipt",
    ]

    missing = st.multiselect(
        "Force-remove channel classes",
        all_channels,
    )

# ---------------------------------------------------------------------
# Generate scenario and retained evidence
# ---------------------------------------------------------------------

scenario = generate_scenario(
    f"interactive-{seed}",
    family,
    int(seed),
    hardened=hardened,
)

evidence = retain_evidence(
    scenario,
    retention,
    int(seed) + 1,
)

evidence = [
    event
    for event in evidence
    if event.channel not in missing
]

# ---------------------------------------------------------------------
# Reconstruction methods
# ---------------------------------------------------------------------

methods = [
    "outcome",
    "causal",
    "rule",
    "scm",
    "trace-r",
]

results = {
    method: reconstruct(
        scenario,
        evidence,
        method,
        threshold,
        learned_model=M,
    )
    for method in methods
}

trace_result = results["trace-r"]

# ---------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------

m1, m2, m3, m4, m5 = st.columns(5)

m1.metric(
    "Ground truth (simulator)",
    scenario.responsible,
)

m2.metric(
    "TRACE-R result",
    trace_result["prediction"],
)

m3.metric(
    "Statistical observability",
    f"{trace_result['legal_observability']:.3f}",
)

m4.metric(
    "Effective observability",
    f"{trace_result['effective_observability']:.3f}",
)

m5.metric(
    "Max posterior",
    f"{trace_result['confidence']:.3f}",
)

# ---------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------

tabs = st.tabs(
    [
        "Reconstruction",
        "Adaptive Capacity",
        "Responsibility Code",
        "Optimal Mixed Code",
        "Signed Receipts",
        "Evidence & DREP",
        "Privacy / Cost",
        "Theory Bounds",
        "Packaged Results",
    ]
)

# =====================================================================
# TAB 1: Reconstruction
# =====================================================================

with tabs[0]:
    st.subheader("Responsibility Reconstruction")

    reconstruction_rows = []

    for method, result in results.items():
        reconstruction_rows.append(
            {
                "method": method,
                "prediction": result["prediction"],
                "confidence": round(result["confidence"], 3),
                "observability": round(
                    result["legal_observability"],
                    3,
                ),
                "effective_observability": round(
                    result["effective_observability"],
                    3,
                ),
                "missing": ", ".join(
                    result["missing_channels"]
                ),
            }
        )

    st.dataframe(
        reconstruction_rows,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("TRACE-R Posterior")

    posterior = trace_result.get("posterior", {})

    if posterior:
        st.bar_chart(posterior)
    else:
        st.info("No posterior distribution available.")

# =====================================================================
# TAB 2: Adaptive Capacity
# =====================================================================

with tabs[1]:
    st.subheader("Adaptive Evidence-Deletion Attack")

    budget = st.slider(
        "Adaptive channel-erasure budget",
        min_value=0,
        max_value=3,
        value=1,
        step=1,
        key="adaptive_budget",
    )

    present_channels = sorted(
        {event.channel for event in scenario.events}
    )

    if budget == 0:
        channel_combinations = [()]
    else:
        effective_budget = min(
            budget,
            len(present_channels),
        )

        channel_combinations = list(
            combinations(
                present_channels,
                effective_budget,
            )
        )

    candidates = []

    for removed_channels in channel_combinations:
        attacked_evidence = [
            event
            for event in scenario.events
            if event.channel not in removed_channels
        ]

        attacked_result = reconstruct(
            scenario,
            attacked_evidence,
            "trace-r",
            threshold,
            learned_model=M,
        )

        candidates.append(
            (
                attacked_result["effective_observability"],
                removed_channels,
                attacked_result,
            )
        )

    candidates.sort(
        key=lambda item: (
            item[0],
            item[2]["confidence"],
        )
    )

    worst_observability, worst_channels, worst_result = candidates[0]

    st.metric(
        "Worst-case responsibility capacity",
        f"{worst_observability:.3f}",
    )

    st.write(
        "**Adaptive removal:**",
        ", ".join(worst_channels)
        if worst_channels
        else "None",
    )

    st.write(
        "**TRACE-R result after attack:**",
        worst_result["prediction"],
    )

    st.write(
        "**Missing evidentiary predicates:**",
        ", ".join(worst_result["missing_channels"])
        if worst_result["missing_channels"]
        else "None",
    )

# =====================================================================
# TAB 3: Responsibility Code
# =====================================================================

with tabs[2]:
    st.subheader("Responsibility Code Construction")

    k = st.slider(
        "Responsibility states",
        min_value=2,
        max_value=12,
        value=7,
        key="responsibility_states",
    )

    redundancy = st.slider(
        "Independent repetition / witness redundancy",
        min_value=1,
        max_value=6,
        value=2,
        key="responsibility_redundancy",
    )

    codebook = repetition_codebook(
        [f"H{i + 1}" for i in range(k)],
        redundancy,
    )

    summary = summarize_codebook(codebook)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Code channels",
        summary.channels,
    )

    c2.metric(
        "Responsibility distance",
        summary.distance,
    )

    c3.metric(
        "Guaranteed erasures",
        summary.erasures_correctable,
    )

    c4.metric(
        "Correctable corruptions",
        summary.corruptions_correctable,
    )

    st.metric(
        "Responsibility code rate",
        f"{summary.rate:.3f}",
    )

    max_erasure_slider = min(
        5,
        summary.channels,
    )

    test_erasure_budget = st.slider(
        "Test erasure budget",
        min_value=0,
        max_value=max_erasure_slider,
        value=min(
            1,
            max_erasure_slider,
        ),
        key="test_erasure_budget",
    )

    if test_erasure_budget <= 3:
        guarantee = erasure_guarantee_holds(
            codebook,
            test_erasure_budget,
        )
    else:
        guarantee = (
            "Skipped in the interactive app because exhaustive "
            "enumeration becomes combinatorially expensive."
        )

    st.write(
        "**Exhaustive unique-decoding check:**",
        guarantee,
    )

    st.latex(
        r"d_R^{\min}="
        r"\min_{i\ne j}d_H(c(H_i),c(H_j)),"
        r"\qquad b<d_R^{\min}"
    )

# =====================================================================
# TAB 4: Optimal Mixed Code
# =====================================================================

with tabs[3]:
    st.subheader(
        "Minimum-Cost Mixed Error-Erasure Responsibility Code"
    )

    kopt = st.slider(
        "States for optimized code",
        min_value=2,
        max_value=10,
        value=7,
        key="kopt",
    )

    bopt = st.slider(
        "Erasure budget b",
        min_value=0,
        max_value=5,
        value=2,
        key="bopt",
    )

    topt = st.slider(
        "Corruption budget t",
        min_value=0,
        max_value=3,
        value=1,
        key="topt",
    )

    answer = optimize_responsibility_code(
        [f"H{i + 1}" for i in range(kopt)],
        erasures=bopt,
        corruptions=topt,
    )

    if answer["success"]:
        st.latex(
            r"2t+b<d_R^{\min}"
        )

        d1, d2, d3, d4 = st.columns(4)

        d1.metric(
            "Required distance",
            answer["target_distance"],
        )

        d2.metric(
            "Optimized channels",
            answer["channels"],
        )

        d3.metric(
            "Achieved distance",
            answer["distance"],
        )

        d4.metric(
            "Code rate",
            f"{answer['rate']:.3f}",
        )

        st.metric(
            "Normalized design cost",
            f"{answer['cost']:.3f}",
        )

        st.write(
            "**Mixed error / erasure guarantee:**",
            answer["guarantee"],
        )

        st.write(
            "**Selected independent claims:**",
            ", ".join(
                answer["selected_claims"]
            ),
        )

    else:
        st.error(
            answer["message"]
        )

# =====================================================================
# TAB 5: Signed Receipts
# =====================================================================

with tabs[4]:
    st.subheader(
        "Ed25519 Cross-Domain Responsibility Receipts"
    )

    identity = SigningIdentity.generate(
        "agent-A"
    )

    receipt = signed_cross_domain_receipt(
        "event-digest",
        "authorization",
        identity,
        "service-B",
        1,
    )

    st.write(
        "**Ed25519 verification:**",
        verify_signed_record(receipt),
    )

    st.json(
        receipt,
        expanded=False,
    )

    tampered = dict(receipt)
    tampered["claim"] = "permission"

    st.write(
        "**Tampered record verifies:**",
        verify_signed_record(tampered),
    )

    replay_guard = ReplayGuard()

    first_acceptance = replay_guard.accept(
        receipt
    )

    second_acceptance = replay_guard.accept(
        receipt
    )

    st.write(
        "**Replay guard first acceptance:**",
        first_acceptance,
    )

    st.write(
        "**Replay guard second acceptance:**",
        second_acceptance,
    )

    st.write(
        "**Merkle root:**",
        merkle_root([receipt]),
    )

# =====================================================================
# TAB 6: Evidence and DREP
# =====================================================================

with tabs[5]:
    st.subheader(
        "Preserved Evidence"
    )

    evidence_rows = []

    for event in evidence:
        evidence_rows.append(
            {
                "t": event.t,
                "actor": event.actor,
                "kind": event.kind,
                "target": event.target,
                "value": event.value,
                "channel": event.channel,
                "digest": (
                    event.digest[:14] + "..."
                ),
            }
        )

    st.dataframe(
        evidence_rows,
        use_container_width=True,
        hide_index=True,
    )

    st.write(
        "**Original full chain integrity:**",
        verify_chain(
            scenario.events
        ),
    )

    drep = build_drep(
        scenario,
        evidence,
        trace_result,
    )

    st.subheader(
        "Digital Responsibility Evidence Package"
    )

    st.json(
        drep,
        expanded=False,
    )

    st.download_button(
        label="Download DREP JSON",
        data=json.dumps(
            drep,
            indent=2,
        ),
        file_name=(
            f"drep_{family}_{seed}.json"
        ),
        mime="application/json",
    )

# =====================================================================
# TAB 7: Privacy / Cost
# =====================================================================

with tabs[6]:
    st.subheader(
        "Telemetry Cost and Privacy-Observability Design"
    )

    default_channels = sorted(
        {
            event.channel
            for event in evidence
        }
    )

    selected = st.multiselect(
        "Telemetry selection",
        all_channels,
        default=default_channels,
        key="telemetry_selection",
    )

    cost = channel_cost(
        selected
    )

    a, bcol, ccol = st.columns(3)

    a.metric(
        "Privacy cost",
        f"{cost['privacy']:.2f}",
    )

    bcol.metric(
        "Storage cost",
        f"{cost['storage']:.2f}",
    )

    ccol.metric(
        "Latency cost",
        f"{cost['latency']:.2f}",
    )

    st.caption(
        "These are normalized research costs used to illustrate "
        "the optimization framework; they are not empirical legal, "
        "financial, or privacy-harm valuations."
    )

# =====================================================================
# TAB 8: Theory Bounds
# =====================================================================

with tabs[7]:
    st.subheader(
        "Information-Theoretic Responsibility Bounds"
    )

    k2 = st.slider(
        "Hypotheses K",
        min_value=2,
        max_value=20,
        value=6,
        key="theory_k",
    )

    mi = st.slider(
        "Mutual information I(H;E), bits",
        min_value=0.0,
        max_value=5.0,
        value=1.0,
        step=0.05,
        key="mutual_information",
    )

    fano_bound = fano_error_lower_bound(
        mi,
        k2,
    )

    st.metric(
        "Fano lower bound",
        f"{fano_bound:.3f}",
    )

    st.latex(
        r"P_e \geq "
        r"\max\left\{0,"
        r"1-\frac{I(H;E)+1}{\log_2 K}"
        r"\right\}"
    )

    st.divider()

    t_theory = st.slider(
        "Corruptions t",
        min_value=0,
        max_value=4,
        value=1,
        key="theory_t",
    )

    b_theory = st.slider(
        "Erasures b",
        min_value=0,
        max_value=6,
        value=1,
        key="theory_b",
    )

    d_theory = st.slider(
        "Responsibility distance d",
        min_value=1,
        max_value=15,
        value=4,
        key="theory_d",
    )

    mixed_ok = mixed_error_erasure_guarantee(
        d_theory,
        t_theory,
        b_theory,
    )

    st.write(
        "**Mixed unique-decoding condition satisfied:**",
        mixed_ok,
    )

    st.latex(
        r"2t+b<d_R^{\min}"
    )

# =====================================================================
# TAB 9: Packaged Results
# =====================================================================

with tabs[8]:
    st.subheader(
        "Reproduced Research Results"
    )

    st.caption(
        "These files correspond to the synchronized result artifacts "
        "used by the manuscript. Machine-dependent cryptographic "
        "microbenchmarks should be interpreted as local measurements."
    )

    result_files = [
        "optimal_responsibility_codes.csv",
        "identifiability_interval.csv",
        "responsibility_transferability.csv",
        "false_attribution_nonidentifiable.csv",
        "adaptive_adversary.csv",
        "risk_coverage_selective.csv",
        "crypto_overhead.csv",
    ]

    result_directory = ROOT / "results"

    if not result_directory.exists():
        st.warning(
            "The results/ directory is not present in this deployment."
        )

    for filename in result_files:
        file_path = result_directory / filename

        if file_path.exists():
            st.subheader(
                filename
            )

            try:
                dataframe = pd.read_csv(
                    file_path
                )

                st.dataframe(
                    dataframe,
                    use_container_width=True,
                    hide_index=True,
                )

            except Exception as exc:
                st.error(
                    f"Could not read {filename}: {exc}"
                )

        else:
            st.info(
                f"{filename} is not included in this deployment."
            )

# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------

st.divider()

st.caption(
    "TRACE-R is a research implementation for studying responsibility "
    "identifiability, evidentiary resilience, and explicit abstention. "
    "It does not determine legal liability or evidentiary admissibility."
)
