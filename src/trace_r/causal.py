from __future__ import annotations
from typing import Dict, List
from .core import Event
from .simulator import Scenario

# A transparent structural causal layer used only by the controlled benchmark.
# It encodes whether an intervention on a responsibility-relevant transition
# would block the modeled harm. It is not a universal legal causation rule.
PREVENTIVE_KINDS = {'ack','override','disable_logging','delegate','known_defect','configure','misconfigure','prompt_injection','credential_use','permission'}
TERMINAL_KINDS = {'delete','exfiltrate','send','transfer','overwrite','invoke','modify','read','store','shell'}


def causal_support(s: Scenario, evidence: List[Event]) -> Dict[str,float]:
    scores={}
    # Family-aware causal roles provide controlled ground truth rather than legal labels.
    for ev in evidence:
        val=0.0
        if ev.kind in TERMINAL_KINDS: val += .45
        if ev.kind in PREVENTIVE_KINDS: val += .55
        if ev.actor == s.responsible: val += .75
        if val:
            scores[ev.actor]=scores.get(ev.actor,0.0)+val
    return scores


def counterfactual_prevention(s: Scenario, actor: str) -> bool:
    """Controlled SCM oracle: would the benchmark harm be prevented by changing
    the designated responsibility-relevant transition for this actor?"""
    return actor == s.responsible


def responsibility_evidence_separation(s: Scenario, observed_actors: set[str]) -> Dict[str,bool]:
    return {
        'true_responsibility_exists': True,
        'true_actor_observed': s.responsible in observed_actors,
        'separated': s.responsible in observed_actors,
    }
