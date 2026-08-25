from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import BernoulliNB
from .core import Event
from .simulator import FAMILIES, generate_scenario, retain_evidence


def event_features(events: List[Event]) -> Dict[str,float]:
    f={}
    channels=set()
    for e in events:
        channels.add(e.channel)
        f[f'kind={e.kind}']=1.0
        f[f'actor={e.actor}|kind={e.kind}']=1.0
        f[f'channel={e.channel}|actor={e.actor}']=1.0
        f[f'channel={e.channel}|kind={e.kind}']=1.0
    for c in channels: f[f'channel={c}']=1.0
    return f

@dataclass
class EvidenceModel:
    vectorizer: DictVectorizer
    model: BernoulliNB
    classes_: List[str]

    def posterior(self, events: List[Event], candidates: List[str]) -> np.ndarray:
        X=self.vectorizer.transform([event_features(events)])
        raw=self.model.predict_proba(X)[0]
        pmap={c:float(p) for c,p in zip(self.model.classes_,raw)}
        arr=np.array([pmap.get(c,1e-12) for c in candidates],dtype=float)
        arr=np.maximum(arr,1e-12); arr/=arr.sum(); return arr


def train_evidence_model(seed: int=7301, n_per_family: int=160, retention_levels=(1.0,.95,.9,.8,.7,.6), exclude_family: str|None=None) -> EvidenceModel:
    rows=[]; labels=[]; k=0
    for fi,fam in enumerate(FAMILIES):
        if fam==exclude_family: continue
        for r in range(n_per_family):
            s=generate_scenario(f'train-{fam}-{r}',fam,seed+fi*100000+r)
            for retention in retention_levels:
                ev=retain_evidence(s,retention,seed+900000+k); k+=1
                rows.append(event_features(ev)); labels.append(s.responsible)
    vec=DictVectorizer(sparse=True)
    X=vec.fit_transform(rows)
    m=BernoulliNB(alpha=.35, fit_prior=True)
    m.fit(X,labels)
    return EvidenceModel(vec,m,list(m.classes_))
