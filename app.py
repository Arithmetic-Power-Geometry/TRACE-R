# Copyright (C) 2026 Mohammad Amir Khusru Akhtar
# Licensed under the Apache License, Version 2.0.
import json
from itertools import combinations
from pathlib import Path
try:
    import streamlit as st
except ImportError:
    raise SystemExit('Install requirements first: pip install -r requirements.txt')

from trace_r.simulator import FAMILIES, generate_scenario, retain_evidence
from trace_r.reconstruct import reconstruct
from trace_r.learned import train_evidence_model
from trace_r.core import verify_chain
from trace_r.evidence import build_drep
from trace_r.theory import channel_cost, fano_error_lower_bound
from trace_r.coding import repetition_codebook, summarize_codebook, erasure_guarantee_holds, optimize_responsibility_code, mixed_error_erasure_guarantee
from trace_r.crypto import SigningIdentity, signed_cross_domain_receipt, verify_signed_record, ReplayGuard, merkle_root

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='TRACE-R Responsibility Coding Lab',layout='wide')
st.title('TRACE-R: Responsibility Coding, Resilience & Legal Observability Lab')
st.caption('Research demonstrator for responsibility-identifiable system design. It reconstructs and tests preserved evidence; it does not decide legal liability or court admissibility.')

@st.cache_resource
def model(): return train_evidence_model(seed=7301,n_per_family=55)
M=model()

with st.sidebar:
    st.header('Scenario controls')
    family=st.selectbox('Incident family',FAMILIES)
    hardened=st.toggle('Witness-hardened evidence architecture',False)
    retention=st.slider('Random evidence retention',0.1,1.0,0.9,0.05)
    threshold=st.slider('Abstention threshold',0.10,0.95,0.58,0.01)
    seed=st.number_input('Seed',0,1000000,42)
    all_channels=['authorization','delegation','permission','action','warning','intervention','policy','identity','model_version','receipt']
    missing=st.multiselect('Force-remove channel classes',all_channels)

s=generate_scenario(f'interactive-{seed}',family,int(seed),hardened=hardened)
ev=retain_evidence(s,retention,int(seed)+1); ev=[e for e in ev if e.channel not in missing]
methods=['outcome','causal','rule','scm','trace-r']
results={m:reconstruct(s,ev,m,threshold,learned_model=M) for m in methods}; r=results['trace-r']

m1,m2,m3,m4,m5=st.columns(5)
m1.metric('Ground truth (simulator)',s.responsible); m2.metric('TRACE-R result',r['prediction'])
m3.metric('Statistical observability',f"{r['legal_observability']:.3f}"); m4.metric('Effective observability',f"{r['effective_observability']:.3f}"); m5.metric('Max posterior',f"{r['confidence']:.3f}")

tabs=st.tabs(['Reconstruction','Adaptive capacity','Responsibility code','Optimal mixed code','Signed receipts','Evidence & DREP','Privacy/cost','Theory bounds','Packaged results'])
with tabs[0]:
    st.dataframe([{'method':m,'prediction':x['prediction'],'confidence':round(x['confidence'],3),'observability':round(x['legal_observability'],3),'effective_observability':round(x['effective_observability'],3),'missing':', '.join(x['missing_channels'])} for m,x in results.items()],use_container_width=True)
    st.bar_chart(r['posterior'])
with tabs[1]:
    budget=st.slider('Adaptive channel-erasure budget',0,3,1,1)
    present=sorted({e.channel for e in s.events}); combos=[()] if budget==0 else list(combinations(present,min(budget,len(present))))
    candidates=[]
    for C in combos:
        attacked=[e for e in s.events if e.channel not in C]; rr=reconstruct(s,attacked,'trace-r',threshold,learned_model=M); candidates.append((rr['effective_observability'],C,rr))
    candidates.sort(key=lambda z:(z[0],z[2]['confidence'])); worst=candidates[0]
    st.metric('Worst-case responsibility capacity',f'{worst[0]:.3f}'); st.write('Adaptive removal:',', '.join(worst[1]) or 'None'); st.write('Result:',worst[2]['prediction'])
with tabs[2]:
    k=st.slider('Responsibility states',2,12,7); redundancy=st.slider('Independent repetition / witness redundancy',1,6,2)
    cb=repetition_codebook([f'H{i+1}' for i in range(k)],redundancy); sm=summarize_codebook(cb)
    c1,c2,c3,c4=st.columns(4); c1.metric('Code channels',sm.channels); c2.metric('Responsibility distance',sm.distance); c3.metric('Guaranteed erasures',sm.erasures_correctable); c4.metric('Correctable corruptions',sm.corruptions_correctable)
    st.metric('Responsibility code rate',f'{sm.rate:.3f}')
    b=st.slider('Test erasure budget',0,min(5,sm.channels),min(1,sm.channels)); st.write('Exhaustive unique-decoding check:',erasure_guarantee_holds(cb,b) if b<=3 else 'Skipped in app for combinatorial cost')
    st.latex(r'd_R=\min_{i\ne j} d_H(c(H_i),c(H_j)),\qquad b<d_R')
with tabs[3]:
    kopt=st.slider('States for optimized code',2,10,7,key='kopt'); bopt=st.slider('Erasure budget b',0,5,2,key='bopt'); topt=st.slider('Corruption budget t',0,3,1,key='topt')
    ans=optimize_responsibility_code([f'H{i+1}' for i in range(kopt)],erasures=bopt,corruptions=topt)
    if ans['success']:
        st.latex(r'2t+b<d_R^{\min}')
        d1,d2,d3,d4=st.columns(4); d1.metric('Required distance',ans['target_distance']); d2.metric('Optimized channels',ans['channels']); d3.metric('Achieved distance',ans['distance']); d4.metric('Code rate',f"{ans['rate']:.3f}")
        st.metric('Normalized design cost',f"{ans['cost']:.3f}"); st.write('Mixed error/erasure guarantee:',ans['guarantee']); st.write('Selected independent claims:',', '.join(ans['selected_claims']))
    else: st.error(ans['message'])
with tabs[4]:
    ident=SigningIdentity.generate('agent-A'); rec=signed_cross_domain_receipt('event-digest','authorization',ident,'service-B',1)
    st.write('Ed25519 verification:',verify_signed_record(rec)); st.json(rec,expanded=False)
    tampered=dict(rec); tampered['claim']='permission'; st.write('Tampered record verifies:',verify_signed_record(tampered))
    guard=ReplayGuard(); first=guard.accept(rec); second=guard.accept(rec); st.write('Replay guard first/second acceptance:',first,second)
    st.write('Merkle root:',merkle_root([rec]))
with tabs[5]:
    st.dataframe([{'t':e.t,'actor':e.actor,'kind':e.kind,'target':e.target,'value':e.value,'channel':e.channel,'digest':e.digest[:14]+'...'} for e in ev],use_container_width=True)
    st.write('Original full chain integrity:',verify_chain(s.events)); drep=build_drep(s,ev,r); st.json(drep,expanded=False)
    st.download_button('Download DREP JSON',json.dumps(drep,indent=2),file_name=f'drep_{family}_{seed}.json',mime='application/json')
with tabs[6]:
    selected=st.multiselect('Telemetry selection',all_channels,default=sorted({e.channel for e in ev})); c=channel_cost(selected)
    a,b,c3=st.columns(3); a.metric('Privacy cost',f"{c['privacy']:.2f}"); b.metric('Storage cost',f"{c['storage']:.2f}"); c3.metric('Latency cost',f"{c['latency']:.2f}")
with tabs[7]:
    k2=st.slider('Hypotheses K',2,20,6); mi=st.slider('Mutual information I(H;E), bits',0.0,5.0,1.0,0.05)
    st.metric('Fano lower bound',f'{fano_error_lower_bound(mi,k2):.3f}'); st.latex(r'P_e \ge 1 - \frac{I(H;E)+1}{\log_2 K}')
    t=st.slider('Corruptions t',0,4,1); b=st.slider('Erasures b',0,6,1); d=st.slider('Responsibility distance d',1,15,4)
    st.write('Mixed unique-decoding condition satisfied:',mixed_error_erasure_guarantee(d,t,b)); st.latex(r'2t+b<d_R^{\min}')
with tabs[8]:
    for fname in ['optimal_responsibility_codes.csv','identifiability_interval.csv','responsibility_transferability.csv','false_attribution_nonidentifiable.csv','adaptive_adversary.csv','risk_coverage_selective.csv','crypto_overhead_signed.csv']:
        p=ROOT/'results'/fname
        if p.exists():
            st.subheader(fname); st.dataframe(__import__('pandas').read_csv(p),use_container_width=True)
