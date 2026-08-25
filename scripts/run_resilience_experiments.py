from __future__ import annotations
import json, math, sys
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

from trace_r.coding import (
    repetition_codebook, summarize_codebook, erasure_guarantee_holds,
    mixed_error_erasure_guarantee, optimize_responsibility_code
)
from trace_r.crypto import SigningIdentity, signed_cross_domain_receipt, verify_signed_record, ReplayGuard, benchmark_crypto
from trace_r.simulator import FAMILIES, generate_scenario, retain_evidence
from trace_r.learned import train_evidence_model
from trace_r.reconstruct import reconstruct
from trace_r.external import export_external_manifest
from trace_r.theory import (
    fano_error_lower_bound, jensen_shannon_divergence,
    bhattacharyya_multiclass_upper_bound, exact_bayes_error_discrete
)

R=ROOT/'results'; F=ROOT/'figures'; R.mkdir(exist_ok=True); F.mkdir(exist_ok=True)
SEED=20260825
CANDIDATES=['developer','provider','operator','agent_A','agent_B','cloud','external_user']


def optimized_code_experiment():
    labels=CANDIDATES
    rows=[]
    for b in range(0,5):
        for t in range(0,3):
            ans=optimize_responsibility_code(labels,erasures=b,corruptions=t)
            rows.append({
                'erasures_b':b,'corruptions_t':t,'required_distance':2*t+b+1,
                'success':int(ans.get('success',False)),'channels':ans.get('channels',np.nan),
                'distance':ans.get('distance',np.nan),'code_rate':ans.get('rate',np.nan),
                'normalized_cost':ans.get('cost',np.nan),'guarantee':int(ans.get('guarantee',False)) if ans.get('success') else 0,
                'selected_claims':'|'.join(ans.get('selected_claims',[]))
            })
    df=pd.DataFrame(rows)
    df.to_csv(R/'optimal_responsibility_codes.csv',index=False)
    return df


def mixed_error_erasure_validation():
    labels=CANDIDATES
    rows=[]
    # Repetition code is used only to validate the general mixed condition on a transparent code.
    for redundancy in range(1,6):
        cb=repetition_codebook(labels,redundancy)
        sm=summarize_codebook(cb)
        for t in range(0,3):
            for b in range(0,5):
                theory=mixed_error_erasure_guarantee(sm.distance,t,b)
                rows.append({'redundancy':redundancy,'distance':sm.distance,'t_corruptions':t,'b_erasures':b,
                             'two_t_plus_b':2*t+b,'theoretical_unique_decoding':int(theory)})
    df=pd.DataFrame(rows); df.to_csv(R/'mixed_error_erasure.csv',index=False); return df


def adaptive_adversary_experiment(n_per_family=12,max_budget=3):
    model=train_evidence_model(seed=SEED+551,n_per_family=55)
    rows=[]
    for hardened in (False,True):
        arch='witness-hardened' if hardened else 'base'
        for fi,fam in enumerate(FAMILIES):
            for i in range(n_per_family):
                s=generate_scenario(f'adapt-{arch}-{fam}-{i}',fam,SEED+fi*10000+i,hardened=hardened)
                present=sorted({e.channel for e in s.events})
                clean=reconstruct(s,s.events,'trace-r',learned_model=model)
                for B in range(max_budget+1):
                    candidates=[()] if B==0 else list(combinations(present,min(B,len(present))))
                    scored=[]
                    for C in candidates:
                        ev=[e for e in s.events if e.channel not in set(C)]
                        rr=reconstruct(s,ev,'trace-r',learned_model=model)
                        # adaptive objective: minimize effective observability, then maximize abstention, then reduce confidence
                        score=(rr['effective_observability'], -int(rr['prediction']=='NOT_IDENTIFIABLE'), rr['confidence'])
                        scored.append((score,C,rr))
                    scored.sort(key=lambda z:z[0])
                    _,C,rr=scored[0]
                    rows.append({'architecture':arch,'family':fam,'sid':s.sid,'budget':B,
                                 'clean_effective_observability':clean['effective_observability'],
                                 'adaptive_capacity':rr['effective_observability'],'attack_channels':'|'.join(C),
                                 'abstention':int(rr['prediction']=='NOT_IDENTIFIABLE'),'confidence':rr['confidence']})
    df=pd.DataFrame(rows); df.to_csv(R/'adaptive_adversary.csv',index=False)
    return df


def _signature(events):
    return ';'.join(sorted({e.channel for e in events}))


def identifiability_interval_experiment(n_per_family=220,levels=(.3,.4,.5,.6,.7,.8,.9,1.0)):
    rows=[]
    for retention in levels:
        labels=[]; sigs=[]
        for fi,fam in enumerate(FAMILIES):
            for i in range(n_per_family):
                s=generate_scenario(f'int-{fam}-{i}',fam,SEED+fi*100000+i)
                ev=retain_evidence(s,retention,SEED+700000+fi*100000+i)
                labels.append(s.responsible); sigs.append(_signature(ev))
        classes=sorted(set(labels)); alphabet=sorted(set(sigs)); ci={c:i for i,c in enumerate(classes)}; si={s:i for i,s in enumerate(alphabet)}
        joint=np.zeros((len(classes),len(alphabet)),dtype=float)
        for y,s in zip(labels,sigs): joint[ci[y],si[s]]+=1
        joint/=joint.sum()
        priors=joint.sum(axis=1); cond=joint/np.maximum(priors[:,None],1e-15)
        bayes=exact_bayes_error_discrete(joint)
        upper=bhattacharyya_multiclass_upper_bound(cond,priors)
        # MI from joint
        pe=joint.sum(axis=0); mi=0.0
        for i in range(joint.shape[0]):
            for j in range(joint.shape[1]):
                p=joint[i,j]
                if p>0: mi += p*math.log2(p/max(priors[i]*pe[j],1e-15))
        H=-sum(p*math.log2(p) for p in priors if p>0)
        fano=fano_error_lower_bound(mi,len(classes))
        rows.append({'retention':retention,'n':len(labels),'hypotheses':len(classes),'entropy_bits':H,
                     'mutual_information_bits':mi,'legal_observability':mi/H if H else 1.0,
                     'fano_lower_bound':fano,'exact_bayes_error':bayes,'bhattacharyya_upper_bound':upper})
    df=pd.DataFrame(rows); df.to_csv(R/'identifiability_interval.csv',index=False); return df


def _js_from_signatures(source_sigs,target_sigs):
    alphabet=sorted(set(source_sigs)|set(target_sigs)); idx={s:i for i,s in enumerate(alphabet)}
    p=np.ones(len(alphabet))*1e-6; q=np.ones(len(alphabet))*1e-6
    for s in source_sigs: p[idx[s]]+=1
    for s in target_sigs: q[idx[s]]+=1
    return jensen_shannon_divergence(p,q,base=2.0)


def responsibility_transferability_experiment(n_train=180,n_test=90,retention=.9):
    # Each held-out family is compared with training evidence from the same responsible actor.
    probe={f:generate_scenario('probe-'+f,f,SEED).responsible for f in FAMILIES}
    actor_fams=defaultdict(list)
    for f,a in probe.items(): actor_fams[a].append(f)
    held=[f for f in FAMILIES if len(actor_fams[probe[f]])>=2]
    rows=[]
    for fi,fam in enumerate(held):
        actor=probe[fam]
        source_fams=[x for x in FAMILIES if x!=fam and probe[x]==actor]
        model=train_evidence_model(seed=SEED+1000+fi,n_per_family=70,exclude_family=fam)
        source_sigs=[]
        for sf_i,sf in enumerate(source_fams):
            for i in range(n_train):
                s=generate_scenario(f'tr-src-{fam}-{sf}-{i}',sf,SEED+1000000+fi*100000+sf_i*1000+i)
                ev=retain_evidence(s,retention,SEED+1200000+fi*100000+sf_i*1000+i)
                source_sigs.append(_signature(ev))
        target_sigs=[]; correct=[]; abst=[]; obs=[]
        for i in range(n_test):
            s=generate_scenario(f'tr-tgt-{fam}-{i}',fam,SEED+1400000+fi*10000+i)
            ev=retain_evidence(s,retention,SEED+1500000+fi*10000+i)
            target_sigs.append(_signature(ev)); rr=reconstruct(s,ev,'trace-r',learned_model=model)
            correct.append(rr['prediction']==s.responsible); abst.append(rr['prediction']=='NOT_IDENTIFIABLE'); obs.append(rr['effective_observability'])
        js=_js_from_signatures(source_sigs,target_sigs); transfer=1-js
        rows.append({'held_out_family':fam,'responsible_actor':actor,'source_families':'|'.join(source_fams),
                     'js_divergence':js,'responsibility_transferability':transfer,
                     'exact_accuracy':float(np.mean(correct)),'abstention':float(np.mean(abst)),
                     'mean_effective_observability':float(np.mean(obs))})
    df=pd.DataFrame(rows)
    if len(df)>=3:
        sp=spearmanr(df.responsibility_transferability,df.exact_accuracy)
        pr=pearsonr(df.responsibility_transferability,df.exact_accuracy)
        meta={'spearman_rho':float(sp.statistic),'spearman_p':float(sp.pvalue),'pearson_r':float(pr.statistic),'pearson_p':float(pr.pvalue),'n_families':len(df)}
    else: meta={'spearman_rho':np.nan,'spearman_p':np.nan,'pearson_r':np.nan,'pearson_p':np.nan,'n_families':len(df)}
    df.to_csv(R/'responsibility_transferability.csv',index=False)
    (R/'responsibility_transferability_summary.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    return df,meta


def identical_outcome_and_far(n=300):
    model=train_evidence_model(seed=SEED+777,n_per_family=65)
    rows=[]; families=['unauthorized_delegation','human_override']
    methods=['outcome','causal','rule','scm','trace-r']
    for i in range(n):
        fam=families[i%2]; s=generate_scenario(f'ioc4-{i}',fam,SEED+i)
        critical='delegation' if fam=='unauthorized_delegation' else 'intervention'
        incomplete=[e for e in s.events if e.channel!=critical]
        for state,ev,identifiable in [('non_identifiable',incomplete,False),('critical_evidence_restored',s.events,True)]:
            for method in methods:
                rr=reconstruct(s,ev,method,learned_model=model)
                rows.append({'pair':i,'family':fam,'condition':state,'identifiable':int(identifiable),'method':method,
                             'truth':s.responsible,'prediction':rr['prediction'],'correct':int(rr['prediction']==s.responsible),
                             'abstain':int(rr['prediction']=='NOT_IDENTIFIABLE')})
    df=pd.DataFrame(rows); df.to_csv(R/'identical_outcome_challenge_full.csv',index=False)
    far=[]
    for method,x in df[df.condition=='non_identifiable'].groupby('method'):
        far.append({'method':method,'false_attribution_rate_nonidentifiable':float(1-x.abstain.mean()),
                    'abstention':float(x.abstain.mean()),'forced_correct_fraction':float(x.correct.mean())})
    fdf=pd.DataFrame(far); fdf.to_csv(R/'false_attribution_nonidentifiable.csv',index=False)
    return df,fdf


def risk_coverage_and_aurc():
    model=train_evidence_model(seed=SEED+31,n_per_family=65)
    cases=[]; k=0
    for ret in [.5,.6,.7,.8,.9,1.0]:
        for fi,fam in enumerate(FAMILIES):
            for i in range(45):
                s=generate_scenario(f'rc4-{ret}-{fam}-{i}',fam,SEED+fi*1000+i+int(ret*10000))
                ev=retain_evidence(s,ret,SEED+900000+k); k+=1
                post=model.posterior(ev,CANDIDATES); j=int(np.argmax(post)); pred=CANDIDATES[j]; conf=float(post[j])
                claims={e.channel for e in ev}
                for e in ev:
                    if e.channel=='receipt' and str(e.value).startswith('witness:'): claims.add(str(e.value).split(':',1)[1])
                req=set(s.required_channels); frac=len(req&claims)/max(1,len(req)); score=min(conf,frac)
                cases.append((score,pred==s.responsible))
    rows=[]
    for tau in np.round(np.linspace(0,1,41),3):
        chosen=[correct for score,correct in cases if score>=tau]
        cov=len(chosen)/len(cases); risk=(sum(not c for c in chosen)/len(chosen)) if chosen else 0.0
        rows.append({'threshold':tau,'coverage':cov,'selective_risk':risk,'accepted':len(chosen)})
    df=pd.DataFrame(rows).sort_values('coverage')
    aurc=float(np.trapezoid(df.selective_risk.to_numpy(),df.coverage.to_numpy()))
    df['aurc']=aurc; df.to_csv(R/'risk_coverage_selective.csv',index=False)
    return df,aurc


def crypto_and_byzantine():
    crypto=pd.DataFrame([benchmark_crypto(n=1500,payload_bytes=n) for n in [128,256,512,1024]])
    crypto.to_csv(R/'crypto_overhead_signed.csv',index=False)
    rows=[]
    for m in [3,5,7,9,11]:
        for f in range(0,(m//2)+2):
            rows.append({'witness_domains_m':m,'malicious_domains_f':f,'majority_recovery':int(m>=2*f+1),
                         'byzantine_margin':m-2*f})
    byz=pd.DataFrame(rows); byz.to_csv(R/'byzantine_recovery.csv',index=False)
    return crypto,byz


def make_figures(optcodes,interval,transfer,ioc_far,adaptive,risk):
    # minimum cost vs required resilience
    fig,ax=plt.subplots(figsize=(7.2,4.6))
    z=optcodes[optcodes.success==1].copy(); z['attack_weight']=2*z.corruptions_t+z.erasures_b
    a=z.groupby('attack_weight',as_index=False).agg(cost=('normalized_cost','min'),channels=('channels','min'))
    ax.plot(a.attack_weight,a.cost,marker='o'); ax.set_xlabel('Required mixed attack weight 2t+b'); ax.set_ylabel('Minimum normalized code cost'); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(F/'minimum_cost_code.pdf'); fig.savefig(F/'minimum_cost_code.png',dpi=220); plt.close(fig)

    fig,ax=plt.subplots(figsize=(7.2,4.6))
    ax.plot(interval.retention,interval.fano_lower_bound,marker='o',label='Fano lower bound')
    ax.plot(interval.retention,interval.exact_bayes_error,marker='s',label='Empirical MAP error')
    ax.plot(interval.retention,interval.bhattacharyya_upper_bound,marker='^',label='Bhattacharyya upper bound')
    ax.set_xlabel('Evidence retention probability'); ax.set_ylabel('Attribution error'); ax.set_ylim(-.02,1.02); ax.grid(alpha=.25); ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(F/'identifiability_interval.pdf'); fig.savefig(F/'identifiability_interval.png',dpi=220); plt.close(fig)

    fig,ax=plt.subplots(figsize=(7.2,4.6))
    ax.scatter(transfer.responsibility_transferability,transfer.exact_accuracy,s=55)
    short={'unauthorized_delegation':'delegation','credential_misuse':'credential','ignored_warning':'warning','human_override':'override','multi_agent_chain':'multi-agent','log_suppression':'log suppress.'}
    yoffs=[8,-13,9,7,-14,7]
    for idx,(_,r) in enumerate(transfer.iterrows()):
        ax.annotate(short.get(r.held_out_family,r.held_out_family),(r.responsibility_transferability,r.exact_accuracy),fontsize=7,xytext=(5,yoffs[idx%len(yoffs)]),textcoords='offset points')
    ax.set_xlabel('Responsibility transferability (1 - JS divergence)'); ax.set_ylabel('Held-out exact accuracy'); ax.set_xlim(0,0.34); ax.set_ylim(-.04,0.72); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(F/'responsibility_transferability.pdf'); fig.savefig(F/'responsibility_transferability.png',dpi=220); plt.close(fig)

    fig,ax=plt.subplots(figsize=(7.2,4.6))
    q=ioc_far.copy(); labels=[('TRACE-R' if m=='trace-r' else m) for m in q.method]
    ax.bar(labels,q.false_attribution_rate_nonidentifiable); ax.set_ylabel('False attribution rate under non-identifiability'); ax.set_ylim(0,1.05); ax.tick_params(axis='x',rotation=20); fig.tight_layout()
    fig.savefig(F/'false_attribution_nonidentifiable.pdf'); fig.savefig(F/'false_attribution_nonidentifiable.png',dpi=220); plt.close(fig)

    fig,ax=plt.subplots(figsize=(7.2,4.6))
    s=adaptive.groupby(['architecture','budget'],as_index=False).adaptive_capacity.mean()
    for arch,x in s.groupby('architecture'): ax.plot(x.budget,x.adaptive_capacity,marker='o',label=arch)
    ax.set_xlabel('Adaptive erasure budget'); ax.set_ylabel('Worst-case mean effective observability'); ax.set_ylim(-.02,1.02); ax.grid(alpha=.25); ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(F/'adaptive_capacity.pdf'); fig.savefig(F/'adaptive_capacity.png',dpi=220); plt.close(fig)

    fig,ax=plt.subplots(figsize=(7.2,4.6)); ax.plot(risk.coverage,risk.selective_risk,marker='o'); ax.set_xlabel('Coverage'); ax.set_ylabel('Selective risk'); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(F/'risk_coverage_selective.pdf'); fig.savefig(F/'risk_coverage_selective.png',dpi=220); plt.close(fig)


def external_manifest_v4():
    manifest=export_external_manifest(R/'external_dataset_manifest.json')
    manifest.update({
      'MP-Bench': {'source':'https://github.com/adobe-research/multi-agent-eval-bench','reported_logs':289,'reported_annotators_per_instance':3,
                   'purpose':'expert multi-perspective failure-attribution validation'},
      'TraceElephant': {'source':'https://github.com/TraceElephant/TraceElephant','purpose':'fully observable multi-agent failure attribution'},
      'AgenTracer': {'source':'https://arxiv.org/abs/2509.03312','purpose':'counterfactual-replay failure attribution baseline'},
      'CHIEF': {'source':'https://arxiv.org/abs/2602.23701','purpose':'hierarchical causal-graph attribution baseline'},
      'Proof-or-Stop': {'source':'https://arxiv.org/abs/2607.14890','purpose':'evidence-gated lifecycle-control comparison'}
    })
    (R/'external_dataset_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest


def main():
    external_manifest_v4()
    opt=optimized_code_experiment(); mixed=mixed_error_erasure_validation(); adaptive=adaptive_adversary_experiment()
    interval=identifiability_interval_experiment(); transfer,tmeta=responsibility_transferability_experiment()
    ioc,far=identical_outcome_and_far(); risk,aurc=risk_coverage_and_aurc(); crypto,byz=crypto_and_byzantine()
    make_figures(opt,interval,transfer,far,adaptive,risk)
    summary={
      'optimal_code_examples': opt[(opt.erasures_b.isin([0,1,2,3]))&(opt.corruptions_t.isin([0,1]))].to_dict('records'),
      'transferability':tmeta,
      'far_nonidentifiable':far.set_index('method').false_attribution_rate_nonidentifiable.to_dict(),
      'aurc':aurc,
      'identifiability_interval':interval.to_dict('records'),
      'adaptive_capacity':adaptive.groupby(['architecture','budget']).adaptive_capacity.mean().reset_index().to_dict('records'),
      'crypto':crypto.to_dict('records'),
      'byzantine':byz.to_dict('records')
    }
    (R/'resilience_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({'status':'ok','results':str(R),'transferability':tmeta,'aurc':aurc},indent=2))

if __name__=='__main__': main()
