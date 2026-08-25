from __future__ import annotations
import json, math, sys, time
from pathlib import Path
from itertools import combinations
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binomtest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trace_r.coding import repetition_codebook, summarize_codebook, erasure_guarantee_holds
from trace_r.crypto import SigningIdentity, signed_cross_domain_receipt, verify_signed_record, ReplayGuard, benchmark_crypto, merkle_root
from trace_r.simulator import FAMILIES, generate_scenario, retain_evidence
from trace_r.learned import train_evidence_model
from trace_r.reconstruct import reconstruct
from trace_r.external import export_external_manifest

R=ROOT/'results'; F=ROOT/'figures'; R.mkdir(exist_ok=True); F.mkdir(exist_ok=True)
SEED=20260825


def coding_experiment():
    labels=['developer','provider','operator','agent_A','agent_B','cloud','external_user']
    rows=[]
    for r in range(1,6):
        cb=repetition_codebook(labels,r); s=summarize_codebook(cb)
        for b in range(0,min(s.channels,6)):
            # exhaustive verification gets expensive only at larger b; verify through d-1 plus one failure point
            checked=b <= min(s.erasures_correctable+1,2)
            holds=erasure_guarantee_holds(cb,b) if checked else np.nan
            rows.append(dict(redundancy=r,states=s.states,channels=s.channels,distance=s.distance,
                             code_rate=s.rate,erasure_budget=b,theoretical_guarantee=int(b<=s.erasures_correctable),
                             exhaustive_verified=holds,corruptions_correctable=s.corruptions_correctable))
    df=pd.DataFrame(rows); df.to_csv(R/'responsibility_coding.csv',index=False)
    return df


def crypto_experiment():
    sizes=[128,256,512,1024]
    rows=[]
    for nbytes in sizes:
        x=benchmark_crypto(n=1500,payload_bytes=nbytes); rows.append(x)
    df=pd.DataFrame(rows); df.to_csv(R/'crypto_overhead.csv',index=False)
    return df


def deceptive_attack_experiment(n=300):
    rows=[]
    for i in range(n):
        signer=SigningIdentity.generate('domain-A')
        rec=signed_cross_domain_receipt('e'+str(i),'authorization',signer,'domain-B',i,timestamp_ns=1000000+i)
        # authentic
        rows.append({'attack':'none','detected':int(not verify_signed_record(rec)),'accepted':int(verify_signed_record(rec))})
        # substitution
        sub=dict(rec); sub['claim']='permission'
        rows.append({'attack':'substitution','detected':int(not verify_signed_record(sub)),'accepted':int(verify_signed_record(sub))})
        # chronology forgery
        tim=dict(rec); tim['timestamp_ns']=tim['timestamp_ns']-999
        rows.append({'attack':'timestamp_forgery','detected':int(not verify_signed_record(tim)),'accepted':int(verify_signed_record(tim))})
        # signer identity substitution
        ids=dict(rec); ids['signer']='domain-X'
        rows.append({'attack':'identity_substitution','detected':int(not verify_signed_record(ids)),'accepted':int(verify_signed_record(ids))})
        # replay: first accepted, second rejected by replay guard
        g=ReplayGuard(); first=g.accept(rec); second=g.accept(rec)
        rows.append({'attack':'replay','detected':int(first and not second),'accepted':int(second)})
    df=pd.DataFrame(rows); df.to_csv(R/'deceptive_attacks.csv',index=False)
    return df


def byzantine_experiment(trials=2000):
    rng=np.random.default_rng(SEED)
    rows=[]
    for m in [3,5,7,9]:
        for f in range(0,(m//2)+2):
            success=[]
            for _ in range(trials):
                # worst-case f malicious witnesses vote false; honest vote true.
                malicious=min(f,m)
                honest=m-malicious
                success.append(int(honest>malicious))
            rows.append({'witness_domains':m,'malicious_domains':f,'success_rate':float(np.mean(success)),
                         'majority_condition':int(m>=2*f+1)})
    df=pd.DataFrame(rows); df.to_csv(R/'byzantine_resilience.csv',index=False)
    return df


def risk_coverage_experiment():
    # Risk-coverage over a continuous evidentiary sufficiency score.  The score is
    # the minimum of learned posterior confidence and the fraction of benchmark
    # prerequisites preserved.  This does not alter TRACE-R's conservative default;
    # it evaluates the coverage/risk trade-off available to a tunable selector.
    model=train_evidence_model(seed=SEED+31,n_per_family=55)
    cases=[]; k=0
    candidates=['developer','provider','operator','agent_A','agent_B','cloud','external_user']
    for ret in [.6,.7,.8,.9,1.0]:
        for fi,fam in enumerate(FAMILIES):
            for i in range(35):
                s=generate_scenario(f'rc-{ret}-{fam}-{i}',fam,SEED+fi*1000+i+int(ret*10000))
                ev=retain_evidence(s,ret,SEED+900000+k); k+=1
                post=model.posterior(ev,candidates); j=int(np.argmax(post)); pred=candidates[j]; conf=float(post[j])
                claims={e.channel for e in ev}
                for e in ev:
                    if e.channel=='receipt' and str(e.value).startswith('witness:'): claims.add(str(e.value).split(':',1)[1])
                req=set(s.required_channels); frac=len(req & claims)/max(1,len(req))
                score=min(conf,frac)
                cases.append((score,pred==s.responsible))
    rows=[]
    for tau in np.round(np.linspace(0.0,1.0,21),2):
        chosen=[correct for score,correct in cases if score>=tau]
        accepted=len(chosen); errors=sum(not c for c in chosen)
        rows.append({'threshold':tau,'coverage':accepted/len(cases),'selective_risk':errors/accepted if accepted else 0.0,'accepted':accepted,'errors':errors})
    df=pd.DataFrame(rows); df.to_csv(R/'risk_coverage.csv',index=False)
    return df

def bootstrap_and_mcnemar():
    p=R/'benchmark_trials.csv'
    if not p.exists(): return pd.DataFrame()
    df=pd.read_csv(p); x=df[np.isclose(df.retention,.9)].copy()
    wide=x.pivot_table(index='sid',columns='method',values='correct',aggfunc='first').dropna()
    rng=np.random.default_rng(SEED); rows=[]
    for method in sorted(x.method.unique()):
        vals=x[x.method==method].correct.to_numpy(float)
        boots=[rng.choice(vals,size=len(vals),replace=True).mean() for _ in range(2000)]
        rows.append({'method':method,'retention':.9,'accuracy':vals.mean(),'ci_low':np.quantile(boots,.025),'ci_high':np.quantile(boots,.975)})
    pd.DataFrame(rows).to_csv(R/'bootstrap_accuracy_ci.csv',index=False)
    # Exact McNemar binomial test TRACE-R vs rule and SCM on exact attribution (abstention counts wrong)
    tests=[]
    if 'trace-r' in wide:
        for other in ['rule','scm']:
            if other in wide:
                b=int(((wide['trace-r']==1)&(wide[other]==0)).sum())
                c=int(((wide['trace-r']==0)&(wide[other]==1)).sum())
                pv=binomtest(min(b,c),n=b+c,p=.5,alternative='two-sided').pvalue if b+c else 1.0
                tests.append({'comparison':f'trace-r_vs_{other}','trace_only_correct':b,'other_only_correct':c,'p_value':pv})
    tdf=pd.DataFrame(tests); tdf.to_csv(R/'mcnemar_tests.csv',index=False)
    return pd.DataFrame(rows)


def identical_outcome_challenge(n=200):
    model=train_evidence_model(seed=SEED+777,n_per_family=55)
    rows=[]
    families=['unauthorized_delegation','human_override']
    for i in range(n):
        fam=families[i%2]; s=generate_scenario(f'ioc-{i}',fam,SEED+i)
        critical='delegation' if fam=='unauthorized_delegation' else 'intervention'
        incomplete=[e for e in s.events if e.channel!=critical]
        complete=s.events
        for state,ev in [('ordinary_trace',incomplete),('receipt_revealed',complete)]:
            for method in ['outcome','causal','rule','scm','trace-r']:
                rr=reconstruct(s,ev,method,learned_model=model)
                rows.append({'pair':i,'family':fam,'condition':state,'method':method,'truth':s.responsible,
                             'prediction':rr['prediction'],'correct':int(rr['prediction']==s.responsible),
                             'abstain':int(rr['prediction']=='NOT_IDENTIFIABLE')})
    df=pd.DataFrame(rows); df.to_csv(R/'identical_outcome_challenge.csv',index=False)
    return df


def make_figures(coding,crypto,byz,risk,ioc):
    fig,ax=plt.subplots(figsize=(7.2,4.6))
    s=coding.drop_duplicates('redundancy').sort_values('redundancy')
    ax.plot(s.code_rate,s.distance,marker='o')
    ax.set_xlabel('Responsibility code rate'); ax.set_ylabel('Minimum responsibility distance')
    ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(F/'coding_rate_distance.pdf'); fig.savefig(F/'coding_rate_distance.png',dpi=220); plt.close(fig)

    fig,ax=plt.subplots(figsize=(7.2,4.6)); ax.plot(crypto.payload_bytes,crypto.sign_median_ms,marker='o',label='sign')
    ax.plot(crypto.payload_bytes,crypto.verify_median_ms,marker='s',label='verify'); ax.set_xlabel('Payload bytes'); ax.set_ylabel('Median latency (ms)'); ax.grid(alpha=.25); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(F/'crypto_overhead.pdf'); fig.savefig(F/'crypto_overhead.png',dpi=220); plt.close(fig)

    fig,ax=plt.subplots(figsize=(7.2,4.6)); ax.plot(risk.coverage,risk.selective_risk,marker='o'); ax.set_xlabel('Coverage'); ax.set_ylabel('Selective risk'); ax.set_ylim(-.01,max(.05,risk.selective_risk.max()+.02)); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(F/'risk_coverage.pdf'); fig.savefig(F/'risk_coverage.png',dpi=220); plt.close(fig)

    summ=ioc.groupby(['condition','method'],as_index=False).agg(accuracy=('correct','mean'),abstention=('abstain','mean'))
    methods=['outcome','causal','rule','scm','trace-r']; x=np.arange(len(methods)); w=.35
    a=summ[summ.condition=='ordinary_trace'].set_index('method').reindex(methods)
    b=summ[summ.condition=='receipt_revealed'].set_index('method').reindex(methods)
    fig,ax=plt.subplots(figsize=(8,4.8)); ax.bar(x-w/2,a.accuracy,w,label='critical evidence hidden'); ax.bar(x+w/2,b.accuracy,w,label='critical evidence restored')
    ax.set_xticks(x,[('trace-r' if m=='trace-r' else m) for m in methods],rotation=15); ax.set_ylabel('Exact attribution accuracy'); ax.set_ylim(0,1.05); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(F/'identical_outcome_challenge.pdf'); fig.savefig(F/'identical_outcome_challenge.png',dpi=220); plt.close(fig)


def main():
    export_external_manifest(R/'external_dataset_manifest.json')
    coding=coding_experiment(); crypto=crypto_experiment(); deceptive=deceptive_attack_experiment(); byz=byzantine_experiment(); risk=risk_coverage_experiment(); ci=bootstrap_and_mcnemar(); ioc=identical_outcome_challenge(); make_figures(coding,crypto,byz,risk,ioc)
    summary={
      'coding': coding.groupby('redundancy').first().reset_index()[['redundancy','channels','distance','code_rate','corruptions_correctable']].to_dict('records'),
      'crypto': crypto.to_dict('records'),
      'deceptive_detection': deceptive.groupby('attack').detected.mean().to_dict(),
      'risk_coverage': risk.to_dict('records'),
      'identical_outcome': ioc.groupby(['condition','method']).agg(accuracy=('correct','mean'),abstention=('abstain','mean')).reset_index().to_dict('records')
    }
    (R/'extended_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({'status':'ok','results':str(R)},indent=2))

if __name__=='__main__': main()
