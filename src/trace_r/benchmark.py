from __future__ import annotations
from pathlib import Path
from itertools import combinations
from collections import Counter
import math
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from .simulator import FAMILIES, generate_scenario, retain_evidence
from .reconstruct import reconstruct
from .learned import train_evidence_model
from .theory import CHANNELS, channel_cost, pareto_frontier, fano_error_lower_bound

METHODS=['outcome','causal','rule','scm','trace-r']
PRIMARY_CHANNELS=['authorization','delegation','permission','action','warning','intervention','model_version','identity','policy']


def _mi_discrete(labels, signatures):
    n=len(labels); cy=Counter(labels); cs=Counter(signatures); cys=Counter(zip(labels,signatures))
    mi=0.0
    for (y,s),c in cys.items():
        p=c/n; py=cy[y]/n; ps=cs[s]/n
        mi += p*math.log2(p/(py*ps))
    return mi


def run_trials(n_per_family=40, levels=(1.0,.9,.8,.7,.6,.5,.4,.3), seed=20260825):
    model=train_evidence_model(seed=seed+31,n_per_family=50)
    rows=[]; k=0
    for fi,fam in enumerate(FAMILIES):
        for r in range(n_per_family):
            s=generate_scenario(f'{fam}-{r:03d}',fam,seed+fi*10000+r)
            for retention in levels:
                ev=retain_evidence(s,retention,seed+700000+k); k+=1
                for m in METHODS:
                    rr=reconstruct(s,ev,m,learned_model=model)
                    pred=rr['prediction']; correct=int(pred==s.responsible)
                    rows.append(dict(sid=s.sid,family=fam,retention=retention,method=m,
                                     truth=s.responsible,prediction=pred,correct=correct,
                                     abstain=int(pred=='NOT_IDENTIFIABLE'),
                                     confidence=rr['confidence'],legal_observability=rr['legal_observability'],
                                     effective_observability=rr['effective_observability'],
                                     missing_count=len(rr['missing_channels'])))
    return pd.DataFrame(rows)


def summarize(df):
    g=df.groupby(['method','retention'],as_index=False).agg(
        accuracy=('correct','mean'), abstention=('abstain','mean'),
        mean_observability=('legal_observability','mean'),
        mean_effective_observability=('effective_observability','mean'), n=('correct','size'))
    sel=[]
    for (m,r),x in df.groupby(['method','retention']):
        y=x[x.prediction!='NOT_IDENTIFIABLE']
        sel.append((m,r,float(y.correct.mean()) if len(y) else np.nan))
    sdf=pd.DataFrame(sel,columns=['method','retention','selective_accuracy'])
    return g.merge(sdf,on=['method','retention'])


def paired_identifiability(seed=88, n=200):
    model=train_evidence_model(seed=seed+900,n_per_family=45)
    rows=[]
    for i in range(n):
        fam='unauthorized_delegation' if i%2==0 else 'human_override'
        s=generate_scenario(f'pair-{i}',fam,seed+i)
        miss='delegation' if fam=='unauthorized_delegation' else 'intervention'
        ev0=retain_evidence(s,1.0,seed+i,targeted_missing=miss)
        ev1=retain_evidence(s,1.0,seed+i)
        for state,ev in [('critical_channel_removed',ev0),('complete_evidence',ev1)]:
            r=reconstruct(s,ev,'trace-r',learned_model=model)
            rows.append(dict(pair=i,family=fam,state=state,truth=s.responsible,
                             prediction=r['prediction'],correct=int(r['prediction']==s.responsible),
                             abstain=int(r['prediction']=='NOT_IDENTIFIABLE'),observability=r['legal_observability'],
                             effective_observability=r['effective_observability']))
    return pd.DataFrame(rows)


def ablation(seed=99, n_per_channel=50):
    model=train_evidence_model(seed=seed+900,n_per_family=45)
    channels=['authorization','delegation','permission','warning','intervention','policy','identity','model_version']
    rows=[]
    for ch in channels:
        for i in range(n_per_channel):
            fam=FAMILIES[i%len(FAMILIES)]
            s=generate_scenario(f'ab-{ch}-{i}',fam,seed+i)
            ev=retain_evidence(s,1.0,seed+i,targeted_missing=ch)
            rr=reconstruct(s,ev,'trace-r',learned_model=model)
            rows.append(dict(removed=ch,family=fam,truth=s.responsible,prediction=rr['prediction'],
                             correct=int(rr['prediction']==s.responsible),abstain=int(rr['prediction']=='NOT_IDENTIFIABLE'),
                             observability=rr['legal_observability'],effective_observability=rr['effective_observability']))
    return pd.DataFrame(rows)


def adversarial_capacity(seed=771, n_per_family=8, max_budget=2):
    model=train_evidence_model(seed=seed+200,n_per_family=45)
    rows=[]
    for hardened in (False,True):
        arch='witness-hardened' if hardened else 'base'
        for fi,fam in enumerate(FAMILIES):
            for i in range(n_per_family):
                s=generate_scenario(f'cap-{arch}-{fam}-{i}',fam,seed+fi*1000+i,hardened=hardened)
                present=sorted({e.channel for e in s.events})
                clean=reconstruct(s,s.events,'trace-r',learned_model=model)
                for b in range(0,max_budget+1):
                    best=(clean['effective_observability'],(),clean)
                    if b>0:
                        for C in combinations(present,b):
                            ev=[e for e in s.events if e.channel not in C]
                            rr=reconstruct(s,ev,'trace-r',learned_model=model)
                            val=rr['effective_observability']
                            if val < best[0]-1e-12:
                                best=(val,C,rr)
                    rows.append(dict(architecture=arch,family=fam,sid=s.sid,budget=b,
                                     clean_observability=clean['effective_observability'],
                                     capacity=best[0],attack_channels='|'.join(best[1]),
                                     attack_abstention=int(best[2]['prediction']=='NOT_IDENTIFIABLE')))
    return pd.DataFrame(rows)


def cut_set_resilience(seed=121, n_per_family=6, max_size=2):
    model=train_evidence_model(seed=seed+200,n_per_family=45)
    records=[]
    for hardened in (False,True):
        arch='witness-hardened' if hardened else 'base'
        scenarios=[]
        for fi,fam in enumerate(FAMILIES):
            for i in range(n_per_family):
                scenarios.append(generate_scenario(f'cut-{arch}-{fam}-{i}',fam,seed+fi*1000+i,hardened=hardened))
        available=PRIMARY_CHANNELS + (['receipt'] if hardened else [])
        found=[]
        for size in range(1,max_size+1):
            for C in combinations(available,size):
                if any(set(prev).issubset(C) for prev in found): continue
                abst=[]; eff=[]
                for s in scenarios:
                    ev=[e for e in s.events if e.channel not in C]
                    rr=reconstruct(s,ev,'trace-r',learned_model=model)
                    abst.append(rr['prediction']=='NOT_IDENTIFIABLE'); eff.append(rr['effective_observability'])
                ar=float(np.mean(abst)); eo=float(np.mean(eff))
                if ar>=0.95:
                    found.append(C)
                    records.append(dict(architecture=arch,cut_size=size,channels='|'.join(C),
                                        abstention=ar,mean_effective_observability=eo))
        if not found:
            records.append(dict(architecture=arch,cut_size=np.nan,channels='',abstention=0.0,mean_effective_observability=np.nan))
    return pd.DataFrame(records)


def privacy_frontier(seed=441, n_per_family=6):
    model=train_evidence_model(seed=seed+200,n_per_family=45)
    scenarios=[]
    for fi,fam in enumerate(FAMILIES):
        for i in range(n_per_family):
            scenarios.append(generate_scenario(f'pf-{fam}-{i}',fam,seed+fi*1000+i))
    candidates=PRIMARY_CHANNELS
    rows=[]
    for mask in range(1,1<<len(candidates)):
        S={candidates[j] for j in range(len(candidates)) if mask&(1<<j)}
        acc=[]; eff=[]; abst=[]
        for s in scenarios:
            ev=[e for e in s.events if e.channel in S]
            rr=reconstruct(s,ev,'trace-r',learned_model=model)
            acc.append(rr['prediction']==s.responsible); eff.append(rr['effective_observability']); abst.append(rr['prediction']=='NOT_IDENTIFIABLE')
        c=channel_cost(S)
        rows.append(dict(channels='|'.join(sorted(S)),n_channels=len(S),cost=c['scalar'],privacy_cost=c['privacy'],
                         storage_cost=c['storage'],latency_cost=c['latency'],observability=float(np.mean(eff)),
                         accuracy=float(np.mean(acc)),abstention=float(np.mean(abst))))
    front=pareto_frontier(rows,x='cost',y='observability')
    return pd.DataFrame(rows),pd.DataFrame(front)


def leave_one_family_out(seed=338, n_test=40):
    # Actor-covered transfer only: a held-out family is evaluated only when the same
    # responsibility actor occurs in at least one remaining family. This avoids the
    # ill-posed case where the held-out label is absent from training altogether.
    probe={f:generate_scenario('probe-'+f,f,seed).responsible for f in FAMILIES}
    counts=Counter(probe.values())
    held_families=[f for f in FAMILIES if counts[probe[f]]>=2]
    rows=[]
    for fi,held in enumerate(held_families):
        model=train_evidence_model(seed=seed+fi*99,n_per_family=55,exclude_family=held)
        for i in range(n_test):
            s=generate_scenario(f'lofo-{held}-{i}',held,seed+fi*10000+i)
            ev=retain_evidence(s,.9,seed+700000+fi*10000+i)
            rr=reconstruct(s,ev,'trace-r',learned_model=model)
            rows.append(dict(held_out_family=held,truth=s.responsible,prediction=rr['prediction'],
                             correct=int(rr['prediction']==s.responsible),abstain=int(rr['prediction']=='NOT_IDENTIFIABLE'),
                             observability=rr['effective_observability']))
    return pd.DataFrame(rows)


def information_bound(seed=551, n_per_family=100, levels=(.3,.4,.5,.6,.7,.8,.9,1.0)):
    rows=[]
    for retention in levels:
        labels=[]; signatures=[]
        for fi,fam in enumerate(FAMILIES):
            for i in range(n_per_family):
                s=generate_scenario(f'mi-{fam}-{i}',fam,seed+fi*10000+i)
                ev=retain_evidence(s,retention,seed+500000+fi*10000+i)
                # Channel-presence signature intentionally discards event semantics. It measures
                # how much responsibility information is carried by the telemetry architecture itself.
                sig=';'.join(sorted(set(e.channel for e in ev)))
                labels.append(s.responsible); signatures.append(sig)
        mi=_mi_discrete(labels,signatures)
        k=len(set(labels)); h=-sum((c/len(labels))*math.log2(c/len(labels)) for c in Counter(labels).values())
        ol=mi/h if h>0 else 1.0
        # Plug-in optimal decoder on this discrete signature.
        groups={}
        for y,sig in zip(labels,signatures): groups.setdefault(sig,Counter())[y]+=1
        correct=sum(max(c.values()) for c in groups.values())
        plugin_error=1-correct/len(labels)
        bound=fano_error_lower_bound(mi,k)
        rows.append(dict(retention=retention,n=len(labels),hypotheses=k,entropy_bits=h,mutual_information_bits=mi,
                         empirical_legal_observability=ol,fano_error_lower_bound=bound,plugin_decoder_error=plugin_error))
    return pd.DataFrame(rows)


def write_outputs(root: Path):
    results=root/'results'; figs=root/'figures'; results.mkdir(exist_ok=True); figs.mkdir(exist_ok=True)
    df=run_trials(); df.to_csv(results/'benchmark_trials.csv',index=False)
    sm=summarize(df); sm.to_csv(results/'benchmark_summary.csv',index=False)
    pair=paired_identifiability(); pair.to_csv(results/'paired_identifiability.csv',index=False)
    ps=pair.groupby('state',as_index=False).agg(accuracy=('correct','mean'),abstention=('abstain','mean'),observability=('effective_observability','mean'))
    ps.to_csv(results/'paired_summary.csv',index=False)
    ab=ablation(); ab.to_csv(results/'ablation.csv',index=False)
    absum=ab.groupby('removed',as_index=False).agg(accuracy=('correct','mean'),abstention=('abstain','mean'),observability=('effective_observability','mean'))
    absum.to_csv(results/'ablation_summary.csv',index=False)
    cap=adversarial_capacity(); cap.to_csv(results/'responsibility_capacity.csv',index=False)
    capsum=cap.groupby(['architecture','budget'],as_index=False).agg(capacity=('capacity','mean'),attack_abstention=('attack_abstention','mean'))
    capsum.to_csv(results/'responsibility_capacity_summary.csv',index=False)
    cuts=cut_set_resilience(); cuts.to_csv(results/'responsibility_cut_sets.csv',index=False)
    pf,pfront=privacy_frontier(); pf.to_csv(results/'privacy_frontier_all.csv',index=False); pfront.to_csv(results/'privacy_frontier.csv',index=False)
    lofo=leave_one_family_out(); lofo.to_csv(results/'leave_one_family_out.csv',index=False)
    lofos=lofo.groupby('held_out_family',as_index=False).agg(accuracy=('correct','mean'),abstention=('abstain','mean'),observability=('observability','mean'))
    lofos.to_csv(results/'leave_one_family_out_summary.csv',index=False)
    info=information_bound(); info.to_csv(results/'information_bound.csv',index=False)

    # Figure 1: accuracy/abstention vs retention
    fig,ax=plt.subplots(figsize=(7.2,4.6))
    for m,x in sm.groupby('method'):
        ax.plot(x.retention,x.accuracy,marker='o',label=('trace-r' if m=='trace-r' else m))
    ax.set_xlabel('Evidence retention probability'); ax.set_ylabel('Exact attribution accuracy')
    ax.set_ylim(0,1.03); ax.grid(alpha=.25); ax.legend(frameon=False,ncol=2); fig.tight_layout()
    fig.savefig(figs/'accuracy_vs_retention.pdf'); fig.savefig(figs/'accuracy_vs_retention.png',dpi=220); plt.close(fig)

    tr=sm[sm.method=='trace-r'].sort_values('retention')
    fig,ax=plt.subplots(figsize=(7.2,4.6)); ax.plot(tr.retention,tr.abstention,marker='o')
    ax.set_xlabel('Evidence retention probability'); ax.set_ylabel('TRACE-R abstention rate'); ax.set_ylim(0,1.03); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(figs/'abstention_vs_retention.pdf'); fig.savefig(figs/'abstention_vs_retention.png',dpi=220); plt.close(fig)

    fig,ax=plt.subplots(figsize=(7.2,4.6)); xx=np.arange(len(ps)); w=.35
    ax.bar(xx-w/2,ps.accuracy,w,label='accuracy'); ax.bar(xx+w/2,ps.abstention,w,label='abstention')
    ax.set_xticks(xx,ps.state,rotation=10); ax.set_ylim(0,1.03); ax.set_ylabel('Rate'); ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(figs/'paired_identifiability.pdf'); fig.savefig(figs/'paired_identifiability.png',dpi=220); plt.close(fig)

    # Capacity figure: hardening shifts failure from budget 1 to budget 2.
    fig,ax=plt.subplots(figsize=(7.2,4.6))
    for arch,x in capsum.groupby('architecture'):
        ax.plot(x.budget,x.capacity,marker='o',label=arch)
    ax.set_xlabel('Adversarial channel-removal budget'); ax.set_ylabel('Mean responsibility capacity')
    ax.set_xticks(sorted(capsum.budget.unique())); ax.set_ylim(-.02,1.02); ax.grid(alpha=.25); ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(figs/'capacity_vs_attack_budget.pdf'); fig.savefig(figs/'capacity_vs_attack_budget.png',dpi=220); plt.close(fig)

    # Privacy-observability frontier
    fig,ax=plt.subplots(figsize=(7.2,4.6)); ax.scatter(pf.cost,pf.observability,s=10,alpha=.25,label='telemetry subsets')
    if len(pfront): ax.plot(pfront.cost,pfront.observability,marker='o',label='Pareto frontier')
    ax.set_xlabel('Normalized telemetry cost'); ax.set_ylabel('Mean effective legal observability'); ax.set_ylim(-.02,1.02); ax.grid(alpha=.25); ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(figs/'privacy_observability_frontier.pdf'); fig.savefig(figs/'privacy_observability_frontier.png',dpi=220); plt.close(fig)

    fig,ax=plt.subplots(figsize=(7.2,4.6))
    ax.plot(info.retention,info.plugin_decoder_error,marker='o',label='plug-in decoder error')
    ax.plot(info.retention,info.fano_error_lower_bound,marker='s',label='Fano lower bound')
    ax.set_xlabel('Evidence retention probability'); ax.set_ylabel('Attribution error')
    ax.set_ylim(-.02,1.02); ax.grid(alpha=.25); ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(figs/'identifiability_bound.pdf'); fig.savefig(figs/'identifiability_bound.png',dpi=220); plt.close(fig)

    return {'trials':df,'summary':sm,'paired':pair,'ablation':ab,'capacity':cap,'cuts':cuts,'privacy':pf,'privacy_frontier':pfront,'lofo':lofo,'information':info}
