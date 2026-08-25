from __future__ import annotations
from typing import List, Dict
from dataclasses import asdict
from datetime import datetime, timezone
from .core import Event, verify_chain


def build_drep(scenario, preserved: List[Event], reconstruction: Dict) -> Dict:
    """Create a research DREP object without converting inferences into source evidence."""
    source = []
    for ev in preserved:
        source.append({
            'class':'original-simulated-record',
            't':ev.t,'actor':ev.actor,'kind':ev.kind,'target':ev.target,'value':ev.value,
            'channel':ev.channel,'parent':ev.parent,'hash_prev':ev.hash_prev,'digest':ev.digest
        })
    derived = {
        'observed_channels': reconstruction['observed_channels'],
        'missing_required_channels': reconstruction['missing_channels'],
        'full_ground_truth_chain_integrity': verify_chain(scenario.events),
        'preserved_record_count': len(preserved),
        'ground_truth_record_count': len(scenario.events),
    }
    inferred = {
        'prediction': reconstruction['prediction'],
        'confidence': reconstruction['confidence'],
        'legal_observability': reconstruction['legal_observability'],
        'responsibility_posterior': reconstruction['posterior'],
    }
    return {
        'format':'DREP-research',
        'generated_at_utc':datetime.now(timezone.utc).isoformat(),
        'disclaimer':'Research reconstruction package. Inferred propositions are not original evidence and are not a legal determination.',
        'scenario_id':scenario.sid,
        'incident_family':scenario.family,
        'source_evidence':source,
        'derived_evidence':derived,
        'inferred_propositions':inferred,
    }
