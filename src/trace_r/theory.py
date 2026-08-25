from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Sequence, Set, Tuple
import math
import numpy as np

CHANNELS = [
    'authorization','delegation','permission','action','warning',
    'intervention','model_version','identity','timestamp','policy','receipt'
]

# Illustrative engineering costs, normalized to [0,1]. They are explicitly not legal weights.
CHANNEL_COSTS: Dict[str, Dict[str, float]] = {
    'authorization': {'privacy':.35,'storage':.15,'latency':.12},
    'delegation': {'privacy':.45,'storage':.20,'latency':.15},
    'permission': {'privacy':.40,'storage':.18,'latency':.10},
    'action': {'privacy':.55,'storage':.60,'latency':.18},
    'warning': {'privacy':.20,'storage':.16,'latency':.08},
    'intervention': {'privacy':.48,'storage':.22,'latency':.12},
    'model_version': {'privacy':.08,'storage':.08,'latency':.04},
    'identity': {'privacy':.85,'storage':.12,'latency':.10},
    'timestamp': {'privacy':.18,'storage':.08,'latency':.03},
    'policy': {'privacy':.15,'storage':.25,'latency':.09},
    'receipt': {'privacy':.22,'storage':.35,'latency':.16},
}


def fano_error_lower_bound(mutual_information_bits: float, k: int) -> float:
    """Classical coarse Fano lower bound for K>=2 equiprobable hypotheses.

    Pe >= 1 - (I(H;E)+1)/log2(K). The clipped value is reported because
    Fano can be vacuous for small K or high information.
    """
    if k <= 1:
        return 0.0
    return float(max(0.0, 1.0 - (float(mutual_information_bits) + 1.0) / math.log2(k)))


def total_variation_error_bound(p: Sequence[float], q: Sequence[float]) -> float:
    """Bayes error for two equal-prior simple hypotheses from total variation."""
    p=np.asarray(p,dtype=float); q=np.asarray(q,dtype=float)
    p=p/max(p.sum(),1e-15); q=q/max(q.sum(),1e-15)
    tv=0.5*np.abs(p-q).sum()
    return float(0.5*(1.0-tv))


def responsibility_capacity(observabilities: Iterable[float]) -> float:
    vals=list(float(v) for v in observabilities)
    return min(vals) if vals else 0.0


def channel_cost(selection: Iterable[str], weights=(1/3,1/3,1/3)) -> Dict[str,float]:
    s=set(selection)
    priv=sum(CHANNEL_COSTS[c]['privacy'] for c in s)
    storage=sum(CHANNEL_COSTS[c]['storage'] for c in s)
    latency=sum(CHANNEL_COSTS[c]['latency'] for c in s)
    scalar=weights[0]*priv+weights[1]*storage+weights[2]*latency
    return {'privacy':priv,'storage':storage,'latency':latency,'scalar':scalar}


def pareto_frontier(rows: List[Dict], x='cost', y='observability') -> List[Dict]:
    """Keep nondominated points: lower x and higher y are preferred."""
    out=[]
    for i,r in enumerate(rows):
        dominated=False
        for j,s in enumerate(rows):
            if i==j: continue
            if s[x] <= r[x] and s[y] >= r[y] and (s[x] < r[x] or s[y] > r[y]):
                dominated=True; break
        if not dominated: out.append(r)
    return sorted(out,key=lambda r:(r[x],-r[y]))


def minimal_cut_sets(required_profiles: Dict[str, Set[str]], max_size: int|None=None) -> List[Tuple[str,...]]:
    """Channel cut sets that collapse at least two distinct responsibility profiles.

    A profile is the set of channels required to distinguish a generating state.
    Removing C maps each profile to profile\\C. C is a cut if two profiles with
    distinct ground-truth labels become identical. Minimality is inclusion-minimal.
    """
    labels=list(required_profiles)
    all_channels=sorted(set().union(*required_profiles.values()))
    max_size=max_size or len(all_channels)
    cuts=[]
    for size in range(1,min(max_size,len(all_channels))+1):
        for comb in combinations(all_channels,size):
            C=set(comb); seen={}; collapse=False
            for lab in labels:
                sig=frozenset(required_profiles[lab]-C)
                if sig in seen and seen[sig] != lab:
                    collapse=True; break
                seen[sig]=lab
            if collapse and not any(set(prev).issubset(C) for prev in cuts):
                cuts.append(comb)
    return cuts

@dataclass(frozen=True)
class CapacityResult:
    clean_observability: float
    worst_observability: float
    adversarial_budget: int
    attack_channels: Tuple[str,...]
    erasure_cost: int|None


def jensen_shannon_divergence(p: Sequence[float], q: Sequence[float], base: float = 2.0) -> float:
    """Jensen-Shannon divergence; with base 2 it lies in [0,1]."""
    p=np.asarray(p,dtype=float); q=np.asarray(q,dtype=float)
    p=p/max(p.sum(),1e-15); q=q/max(q.sum(),1e-15)
    m=.5*(p+q)
    def kl(a,b):
        mask=a>0
        return float(np.sum(a[mask]*(np.log(a[mask]/np.maximum(b[mask],1e-15))/np.log(base))))
    return float(.5*kl(p,m)+.5*kl(q,m))


def bhattacharyya_multiclass_upper_bound(class_conditionals: np.ndarray, priors: Sequence[float]|None=None) -> float:
    """Union-style Bhattacharyya upper bound on multiclass Bayes error.

    class_conditionals has shape (K, M), each row a distribution over a finite evidence alphabet.
    The bound is sum_{i<j} sqrt(pi_i pi_j) * BC(P_i,P_j), clipped to 1.
    """
    P=np.asarray(class_conditionals,dtype=float)
    P=P/np.maximum(P.sum(axis=1,keepdims=True),1e-15)
    k=P.shape[0]
    pi=np.ones(k)/k if priors is None else np.asarray(priors,dtype=float)
    pi=pi/max(pi.sum(),1e-15)
    total=0.0
    for i,j in combinations(range(k),2):
        bc=float(np.sqrt(P[i]*P[j]).sum())
        total += math.sqrt(float(pi[i]*pi[j]))*bc
    return float(min(1.0,total))


def exact_bayes_error_discrete(joint: np.ndarray) -> float:
    """Exact MAP error for a discrete empirical joint table P(H,E)."""
    J=np.asarray(joint,dtype=float)
    J=J/max(J.sum(),1e-15)
    return float(1.0-np.max(J,axis=0).sum())
