"""Adapters for external agent-trace benchmarks.

External corpora are not redistributed. The adapters download them from their official
public sources when network access is available. This avoids licensing ambiguity and
keeps controlled benchmark results separate from external validation.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, List
import json

WHO_WHEN_HF='Kevin355/Who_and_When'


def load_who_when(cache_dir: str|Path='external_data'):
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError('Install optional dependency: pip install datasets') from e
    cache_dir=str(cache_dir)
    out={}
    for cfg in ['Algorithm-Generated','Hand-Crafted']:
        ds=load_dataset(WHO_WHEN_HF,cfg,split='train',cache_dir=cache_dir)
        out[cfg]=ds
    return out


def who_when_summary(cache_dir: str|Path='external_data') -> List[Dict]:
    ds=load_who_when(cache_dir)
    rows=[]
    for cfg, split in ds.items():
        rows.append({'dataset':'Who&When','configuration':cfg,'rows':len(split),
                     'columns':','.join(split.column_names)})
    return rows


def export_external_manifest(path: str|Path):
    manifest={
      'Who&When': {'source':'https://huggingface.co/datasets/Kevin355/Who_and_When',
                   'expected_total_rows':184,
                   'purpose':'external multi-agent failure-attribution trace validation'},
      'AgentDojo': {'source':'https://github.com/ethz-spylab/agentdojo',
                    'purpose':'external tool-mediated agent security trace validation'},
      'GRADE': {'source':'https://github.com/yzhao062/grade',
                'purpose':'external execution/dependency graph interoperability'}
    }
    Path(path).write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest
