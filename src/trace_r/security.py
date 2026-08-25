from __future__ import annotations
from dataclasses import asdict, replace
from typing import Dict, List, Tuple
import hashlib, json, random
from .core import Event


def cross_party_receipt(ev: Event, counterparty: str, key_hint: str='research-key') -> Dict:
    """Research-only deterministic receipt. It models cross-domain corroboration;
    it is not a production digital signature implementation."""
    body={'event_digest':ev.digest,'counterparty':counterparty,'t':ev.t}
    mac=hashlib.sha256((json.dumps(body,sort_keys=True)+key_hint).encode()).hexdigest()
    return {**body,'receipt_digest':mac,'class':'derived-research-receipt'}


def attack_remove_channels(events: List[Event], channels: Tuple[str,...]) -> List[Event]:
    c=set(channels)
    return [e for e in events if e.channel not in c]


def attack_time_reorder(events: List[Event], seed: int=0) -> List[Event]:
    # Reordering preserved Event objects invalidates chain order verification.
    out=list(events); random.Random(seed).shuffle(out); return out


def attack_identity_ambiguity(events: List[Event]) -> List[Event]:
    # Semantic substitution without recomputing the original digest; integrity verification can detect it.
    out=[]
    for e in events:
        if e.channel=='identity':
            out.append(replace(e, value='ambiguous-principal'))
        else: out.append(e)
    return out
