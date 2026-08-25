from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import random
import numpy as np
from .core import chain_events, Event

ACTORS = ['developer','provider','operator','agent_A','agent_B','cloud']
FAMILIES = [
    'unauthorized_delegation','credential_misuse','ignored_warning','prompt_manipulation',
    'model_defect','policy_defect','cloud_defect','human_override','multi_agent_chain','log_suppression'
]

@dataclass
class Scenario:
    sid: str
    family: str
    events: List[Event]
    responsible: str
    duty_holder: str
    required_channels: List[str]
    harm: str


def _base_rows(resp: str, family: str, rng: random.Random) -> Tuple[List[Dict], List[str], str, str]:
    rows = [
        dict(t=1, actor='operator', kind='authorize', target='agent_A', value='task_scope', channel='authorization'),
        dict(t=2, actor='agent_A', kind='load_model', target='provider', value='v1.3', channel='model_version', parent=1),
        dict(t=3, actor='provider', kind='policy', target='agent_A', value='policy_7', channel='policy', parent=2),
        dict(t=4, actor='operator', kind='identity_bind', target='agent_A', value='op-token', channel='identity', parent=1),
    ]
    req = ['authorization','action','identity']
    duty_holder = resp
    harm = 'protected_resource_compromised'
    if family == 'unauthorized_delegation':
        rows += [
            dict(t=5, actor='agent_A', kind='delegate', target='agent_B', value='write_task', channel='delegation', parent=1),
            dict(t=6, actor='agent_B', kind='permission', target='database', value='write', channel='permission', parent=5),
            dict(t=7, actor='agent_B', kind='delete', target='database', value='records', channel='action', parent=6),
        ]; req += ['delegation','permission']; resp='agent_A'; duty_holder='operator'
    elif family == 'credential_misuse':
        rows += [
            dict(t=5, actor='operator', kind='permission', target='agent_A', value='read_only', channel='permission'),
            dict(t=6, actor='agent_A', kind='credential_use', target='cloud', value='admin_token', channel='identity'),
            dict(t=7, actor='agent_A', kind='exfiltrate', target='dataset', value='rows', channel='action', parent=6),
        ]; req += ['permission']; resp='agent_A'; duty_holder='operator'
    elif family == 'ignored_warning':
        rows += [
            dict(t=5, actor='agent_A', kind='permission', target='database', value='write', channel='permission'),
            dict(t=6, actor='monitor', kind='warning', target='operator', value='high_risk', channel='warning'),
            dict(t=7, actor='operator', kind='ack', target='monitor', value='ignored', channel='intervention', parent=6),
            dict(t=8, actor='agent_A', kind='delete', target='database', value='records', channel='action', parent=5),
        ]; req += ['permission','warning','intervention']; resp='operator'; duty_holder='operator'
    elif family == 'prompt_manipulation':
        rows += [
            dict(t=5, actor='external_user', kind='prompt_injection', target='agent_A', value='override', channel='action'),
            dict(t=6, actor='agent_A', kind='permission', target='email', value='send', channel='permission'),
            dict(t=7, actor='agent_A', kind='send', target='victim', value='phish', channel='action', parent=5),
        ]; req += ['permission','policy']; resp='external_user'; duty_holder='provider'
    elif family == 'model_defect':
        rows += [
            dict(t=5, actor='provider', kind='known_defect', target='model', value='tool_routing_bug', channel='warning'),
            dict(t=6, actor='agent_A', kind='permission', target='payments', value='pay', channel='permission'),
            dict(t=7, actor='agent_A', kind='transfer', target='wrong_account', value='1000', channel='action'),
        ]; req += ['model_version','warning']; resp='provider'; duty_holder='provider'
    elif family == 'policy_defect':
        rows += [
            dict(t=5, actor='developer', kind='configure', target='policy', value='unsafe_allow_all', channel='policy'),
            dict(t=6, actor='agent_A', kind='permission', target='filesystem', value='write', channel='permission'),
            dict(t=7, actor='agent_A', kind='overwrite', target='file', value='critical', channel='action'),
        ]; req += ['policy','permission']; resp='developer'; duty_holder='developer'
    elif family == 'cloud_defect':
        rows += [
            dict(t=5, actor='cloud', kind='misconfigure', target='bucket', value='public', channel='permission'),
            dict(t=6, actor='agent_A', kind='store', target='bucket', value='personal_data', channel='action'),
            dict(t=7, actor='external_user', kind='read', target='bucket', value='personal_data', channel='action'),
        ]; req += ['permission']; resp='cloud'; duty_holder='cloud'
    elif family == 'human_override':
        rows += [
            dict(t=5, actor='provider', kind='policy', target='agent_A', value='deny_shell', channel='policy'),
            dict(t=6, actor='operator', kind='override', target='agent_A', value='allow_shell', channel='intervention'),
            dict(t=7, actor='agent_A', kind='shell', target='server', value='rm_data', channel='action', parent=6),
        ]; req += ['policy','intervention']; resp='operator'; duty_holder='operator'
    elif family == 'multi_agent_chain':
        rows += [
            dict(t=5, actor='agent_A', kind='delegate', target='agent_B', value='subtask', channel='delegation'),
            dict(t=6, actor='operator', kind='permission', target='agent_B', value='external_api', channel='permission'),
            dict(t=7, actor='agent_B', kind='invoke', target='api', value='destructive', channel='action', parent=5),
        ]; req += ['delegation','permission']; resp='operator'; duty_holder='operator'
    else:  # log_suppression
        rows += [
            dict(t=5, actor='operator', kind='permission', target='agent_A', value='write', channel='permission'),
            dict(t=6, actor='operator', kind='disable_logging', target='agent_A', value='audit_off', channel='intervention'),
            dict(t=7, actor='agent_A', kind='modify', target='records', value='harmful', channel='action'),
        ]; req += ['intervention','permission']; resp='operator'; duty_holder='operator'
    for j,r in enumerate(rows): r['t'] = j+1
    return rows, sorted(set(req)), resp, duty_holder


def generate_scenario(sid: str, family: str, seed: int, hardened: bool=False) -> Scenario:
    rng = random.Random(seed)
    rows, req, resp, duty = _base_rows('operator', family, rng)
    if hardened:
        # Independent-witness receipts model redundant cross-domain corroboration.
        # They preserve the semantic claim when the primary channel is lost.
        t=len(rows)+1
        for claim in req:
            if claim in ('action','identity','authorization','delegation','permission','warning','intervention','policy'):
                rows.append(dict(t=t, actor='witness_service', kind='receipt', target=claim,
                                 value='witness:'+claim, channel='receipt', parent=None)); t+=1
    for j,r in enumerate(rows): r['t']=j+1
    return Scenario(sid=sid, family=family, events=chain_events(rows), responsible=resp,
                    duty_holder=duty, required_channels=req, harm='protected_resource_compromised')


def retain_evidence(s: Scenario, retention: float, seed: int, targeted_missing: str|None=None) -> List[Event]:
    rng = random.Random(seed)
    kept = []
    for ev in s.events:
        if targeted_missing and ev.channel == targeted_missing:
            continue
        # outcome action is almost always visible, preserving realistic incident discovery
        p = max(retention, 0.97) if ev.kind in ('delete','exfiltrate','transfer','overwrite','send','invoke','modify','read') else retention
        if rng.random() <= p:
            kept.append(ev)
    return kept
