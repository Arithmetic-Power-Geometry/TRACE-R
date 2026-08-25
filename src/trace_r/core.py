from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import hashlib, json, math
import numpy as np

CRITICAL_CHANNELS = {
    'authorization', 'delegation', 'permission', 'action', 'warning',
    'intervention', 'model_version', 'identity', 'timestamp', 'policy'
}

@dataclass(frozen=True)
class Event:
    t: int
    actor: str
    kind: str
    target: str
    value: str
    channel: str
    parent: Optional[int] = None
    hash_prev: str = ''
    digest: str = ''

    def payload(self) -> Dict:
        d = asdict(self)
        d.pop('digest', None)
        return d


def chain_events(rows: List[Dict]) -> List[Event]:
    out: List[Event] = []
    prev = 'GENESIS'
    for i, row in enumerate(rows):
        payload = {
            't': int(row['t']), 'actor': row['actor'], 'kind': row['kind'],
            'target': row.get('target',''), 'value': row.get('value',''),
            'channel': row['channel'], 'parent': row.get('parent'), 'hash_prev': prev
        }
        raw = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
        digest = hashlib.sha256(raw).hexdigest()
        ev = Event(**payload, digest=digest)
        out.append(ev)
        prev = digest
    return out


def verify_chain(events: List[Event]) -> bool:
    prev = 'GENESIS'
    for ev in events:
        payload = ev.payload(); payload['hash_prev'] = prev
        raw = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
        if hashlib.sha256(raw).hexdigest() != ev.digest:
            return False
        prev = ev.digest
    return True


def entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if len(p) == 0: return 0.0
    return float(-(p * np.log2(p)).sum())


def legal_observability(prior: np.ndarray, posterior: np.ndarray) -> float:
    hp = entropy(prior); hq = entropy(posterior)
    if hp <= 1e-12: return 1.0
    return float(max(0.0, min(1.0, 1.0 - hq / hp)))


def posterior_from_scores(scores: np.ndarray, temperature: float = 0.55) -> np.ndarray:
    z = np.asarray(scores, dtype=float) / max(temperature, 1e-6)
    z -= z.max()
    e = np.exp(z)
    return e / e.sum()
