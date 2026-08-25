from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trace_r.external import load_who_when


def _row_to_dict(row):
    try: return dict(row)
    except Exception: return row


def audit_who_when():
    out=ROOT/'results'; out.mkdir(exist_ok=True)
    ds=load_who_when(ROOT/'external_data')
    rows=[]
    for cfg,split in ds.items():
        for i,row in enumerate(split):
            r=_row_to_dict(row)
            hist=r.get('history') or []
            names=[str(m.get('name','')) for m in hist if isinstance(m,dict)]
            mistake=str(r.get('mistake_agent',''))
            step=r.get('mistake_step',None)
            try: step_int=int(step)
            except Exception: step_int=-1
            rows.append({
                'configuration':cfg,'row':i,'history_steps':len(hist),'distinct_named_agents':len(set(n for n in names if n)),
                'mistake_agent':mistake,'mistake_agent_appears_in_trace':int(bool(mistake) and mistake in names),
                'mistake_step':step_int,'decisive_step_within_trace':int(step_int>=0 and step_int < len(hist)),
                'identity_ablation_effective_observability':0.0,
                'complete_identity_predicate':int(bool(mistake) and mistake in names),
            })
    df=pd.DataFrame(rows); df.to_csv(out/'external_who_when_trace_audit.csv',index=False)
    summary=df.groupby('configuration',as_index=False).agg(
        rows=('row','size'),median_steps=('history_steps','median'),median_agents=('distinct_named_agents','median'),
        responsible_identity_present=('mistake_agent_appears_in_trace','mean'),decisive_step_addressable=('decisive_step_within_trace','mean'))
    summary.to_csv(out/'external_who_when_trace_audit_summary.csv',index=False)
    (out/'external_validation_status.json').write_text(json.dumps({'status':'executed','dataset':'Who&When','rows':len(df),'note':'Trace/evidence interoperability audit; not TRACE-R legal-attribution accuracy.'},indent=2))
    print(summary.to_string(index=False))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--who-when',action='store_true'); a=ap.parse_args()
    if a.who_when: audit_who_when()
    else: ap.error('select --who-when')

if __name__=='__main__': main()
