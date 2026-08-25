from __future__ import annotations
from typing import Dict, List
import numpy as np
from .core import Event, legal_observability, posterior_from_scores
from .simulator import ACTORS, Scenario
from .causal import causal_support

SIGNALS = {
 'developer': {'configure':2.4,'policy':0.4},
 'provider': {'known_defect':2.6,'load_model':0.3,'policy':0.6},
 'operator': {'authorize':0.4,'permission':1.0,'override':2.7,'ack':2.4,'disable_logging':2.8},
 'agent_A': {'delegate':2.4,'credential_use':2.2,'delete':1.4,'exfiltrate':1.5,'send':1.0,'transfer':0.7,'overwrite':0.8,'modify':0.8},
 'agent_B': {'delete':1.5,'invoke':1.4},
 'cloud': {'misconfigure':2.7},
 'external_user': {'prompt_injection':2.8,'read':0.3},
}
CANDIDATES = ACTORS + ['external_user']


def _scores(evidence: List[Event], method: str, scenario: Scenario|None=None) -> np.ndarray:
    scores = np.zeros(len(CANDIDATES), dtype=float)
    idx = {a:i for i,a in enumerate(CANDIDATES)}
    if method=='scm' and scenario is not None:
        cs=causal_support(scenario,evidence)
        for a,v in cs.items():
            if a in idx: scores[idx[a]] += v
        if np.allclose(scores,0): scores += .01
        return scores
    for ev in evidence:
        if ev.actor not in idx: continue
        if method == 'outcome':
            if ev.kind in ('delete','exfiltrate','send','transfer','overwrite','invoke','modify','read'):
                scores[idx[ev.actor]] += 1.0
        elif method == 'causal':
            if ev.kind in ('delete','exfiltrate','send','transfer','overwrite','invoke','modify','prompt_injection','misconfigure'):
                scores[idx[ev.actor]] += 1.4
            if ev.kind in ('delegate','override','disable_logging','known_defect'):
                scores[idx[ev.actor]] += 0.7
        elif method == 'rule':
            scores[idx[ev.actor]] += SIGNALS.get(ev.actor,{}).get(ev.kind,0.15)
        else:
            base = SIGNALS.get(ev.actor,{}).get(ev.kind,0.12)
            if ev.channel in ('delegation','intervention','warning','policy','permission'): base *= 1.25
            if ev.actor=='operator' and ev.kind=='permission' and ev.target=='agent_B': base += 3.2
            if ev.channel == 'identity': base *= 1.05
            scores[idx[ev.actor]] += base
    if np.allclose(scores,0): scores += 0.01
    return scores


def reconstruct(s: Scenario, evidence: List[Event], method: str='trace-r', threshold: float=0.58, learned_model=None) -> Dict:
    observed_channels = {e.channel for e in evidence}
    observed_claims=set(observed_channels)
    for e in evidence:
        if e.channel=='receipt' and e.value.startswith('witness:'):
            observed_claims.add(e.value.split(':',1)[1])
    required = set(s.required_channels)
    missing = sorted(required - observed_claims)
    if method in ('trace-r',) and learned_model is not None:
        posterior=learned_model.posterior(evidence,CANDIDATES)
    else:
        base_method = 'trace-r' if method in ('trace-r',) else method
        scores = _scores(evidence, base_method, s)
        posterior = posterior_from_scores(scores, temperature=0.55 if method in ('trace-r','scm') else 0.8)
    prior = np.ones(len(CANDIDATES))/len(CANDIDATES)
    obs = legal_observability(prior, posterior)
    pred_i = int(np.argmax(posterior)); pred = CANDIDATES[pred_i]
    confidence = float(posterior[pred_i])
    if method in ('trace-r',):
        # Two-layer sufficiency: a posterior cannot substitute for a missing benchmark prerequisite.
        if missing or confidence < threshold:
            pred = 'NOT_IDENTIFIABLE'
    return {
        'prediction': pred, 'confidence': confidence, 'legal_observability': obs,
        'posterior': {a: float(p) for a,p in zip(CANDIDATES,posterior)},
        'missing_channels': missing, 'observed_channels': sorted(observed_channels),
        'observed_claims': sorted(observed_claims),
        'effective_observability': float(obs if not missing else 0.0),
        'method':method,
    }
