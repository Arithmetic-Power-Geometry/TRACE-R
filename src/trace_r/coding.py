from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations, product
from typing import Dict, Iterable, Mapping, Sequence, Tuple, List
import math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

Codeword = Tuple[str, ...]


def hamming_distance(a: Sequence[str], b: Sequence[str]) -> int:
    if len(a) != len(b):
        raise ValueError('codewords must have equal length')
    return sum(x != y for x, y in zip(a, b))


def responsibility_distance(codebook: Mapping[str, Sequence[str]]) -> int:
    labels = list(codebook)
    if len(labels) < 2:
        return 0
    return min(hamming_distance(codebook[a], codebook[b])
               for a, b in combinations(labels, 2))


def code_rate(k: int, m: int, alphabet_size: int = 2) -> float:
    if k <= 0 or m <= 0 or alphabet_size < 2:
        return 0.0
    return math.log(k, alphabet_size) / m


def erase(codeword: Sequence[str], positions: Iterable[int]) -> Codeword:
    pos = set(positions)
    return tuple('?' if i in pos else str(x) for i, x in enumerate(codeword))


def compatible(a: Sequence[str], b: Sequence[str]) -> bool:
    return all(x == '?' or y == '?' or x == y for x, y in zip(a, b))


def uniquely_decodable_after_erasure(codebook: Mapping[str, Sequence[str]], label: str,
                                      erased_positions: Iterable[int]) -> bool:
    obs = erase(codebook[label], erased_positions)
    possible = [lab for lab, cw in codebook.items() if compatible(obs, cw)]
    return possible == [label]


def erasure_guarantee_holds(codebook: Mapping[str, Sequence[str]], budget: int) -> bool:
    if not codebook:
        return True
    m = len(next(iter(codebook.values())))
    for b in range(budget + 1):
        for positions in combinations(range(m), b):
            for label in codebook:
                if not uniquely_decodable_after_erasure(codebook, label, positions):
                    return False
    return True


def corruption_guarantee(distance: int) -> int:
    return max(0, (distance - 1) // 2)


def erasure_guarantee(distance: int) -> int:
    return max(0, distance - 1)


def mixed_error_erasure_guarantee(distance: int, corruptions: int, erasures: int) -> bool:
    """Classical unique-decoding condition for t symbol errors and b erasures."""
    return 2 * int(corruptions) + int(erasures) < int(distance)


@dataclass(frozen=True)
class ResponsibilityCodeSummary:
    states: int
    channels: int
    distance: int
    erasures_correctable: int
    corruptions_correctable: int
    rate: float


def summarize_codebook(codebook: Mapping[str, Sequence[str]]) -> ResponsibilityCodeSummary:
    m = len(next(iter(codebook.values()))) if codebook else 0
    d = responsibility_distance(codebook)
    return ResponsibilityCodeSummary(
        states=len(codebook), channels=m, distance=d,
        erasures_correctable=erasure_guarantee(d),
        corruptions_correctable=corruption_guarantee(d),
        rate=code_rate(len(codebook), m) if m else 0.0,
    )


def repetition_codebook(labels: Sequence[str], redundancy: int) -> Dict[str, Codeword]:
    labels = list(labels)
    out: Dict[str, Codeword] = {}
    for i, lab in enumerate(labels):
        base = ['0'] * len(labels)
        base[i] = '1'
        cw = tuple(x for sym in base for x in [sym] * redundancy)
        out[lab] = cw
    return out


def candidate_binary_claims(labels: Sequence[str]) -> List[Tuple[str, Tuple[str, ...], float]]:
    """Unique non-trivial binary partitions, with an illustrative privacy-sensitive cost.

    Complement-equivalent partitions are deduplicated by fixing the first state to 0.
    More singleton-like partitions receive a modest higher privacy cost because they
    isolate a smaller class of principals. Costs are engineering research parameters,
    not legal or empirical privacy valuations.
    """
    labels = list(labels); k = len(labels)
    if k < 2:
        return []
    cols=[]
    for tail in product('01', repeat=k-1):
        col=('0',)+tuple(tail)
        ones=sum(x=='1' for x in col)
        if ones in (0,k):
            continue
        minority=min(ones,k-ones)
        privacy_penalty=(k/2-minority)/(k/2)
        cost=1.0 + 0.35*privacy_penalty
        name='claim_' + ''.join(col)
        cols.append((name,col,float(cost)))
    return cols


def optimize_responsibility_code(labels: Sequence[str], erasures: int=0, corruptions: int=0,
                                 target_distance: int|None=None) -> Dict:
    """Minimum-cost binary evidence architecture using a set-multicover MILP.

    Every selected coordinate is interpreted as an independently controlled,
    authenticated responsibility-bearing claim. Pairwise constraints require each
    state pair to be separated in at least D coordinates, where D defaults to
    2t+b+1 for t corruptions and b erasures.
    """
    labels=list(labels); k=len(labels)
    D=int(target_distance if target_distance is not None else 2*corruptions+erasures+1)
    candidates=candidate_binary_claims(labels)
    n=len(candidates)
    if k < 2 or n == 0:
        raise ValueError('at least two labels are required')
    pairs=list(combinations(range(k),2))
    A=np.zeros((len(pairs),n),dtype=float)
    for r,(i,j) in enumerate(pairs):
        for c,(_,col,_) in enumerate(candidates):
            A[r,c]=1.0 if col[i]!=col[j] else 0.0
    cvec=np.array([x[2] for x in candidates],dtype=float)
    constraint=LinearConstraint(A, lb=np.full(len(pairs),D,dtype=float), ub=np.full(len(pairs),np.inf))
    res=milp(c=cvec, integrality=np.ones(n), bounds=Bounds(np.zeros(n),np.ones(n)), constraints=constraint,
             options={'time_limit':30.0})
    if not res.success or res.x is None:
        return {'success':False,'message':res.message,'target_distance':D}
    selected=[candidates[i] for i,x in enumerate(res.x) if x>0.5]
    codebook={lab:tuple(col[idx] for _,col,_ in selected) for idx,lab in enumerate(labels)}
    summ=summarize_codebook(codebook)
    return {
        'success':True,'message':res.message,'target_distance':D,'erasures':erasures,'corruptions':corruptions,
        'selected_claims':[name for name,_,_ in selected],'cost':float(sum(cost for _,_,cost in selected)),
        'codebook':codebook,'channels':summ.channels,'distance':summ.distance,'rate':summ.rate,
        'guarantee':mixed_error_erasure_guarantee(summ.distance,corruptions,erasures)
    }
