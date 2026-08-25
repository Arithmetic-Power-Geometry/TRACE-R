from trace_r.simulator import generate_scenario
from trace_r.core import verify_chain
from trace_r.reconstruct import reconstruct
from trace_r.theory import fano_error_lower_bound, channel_cost


def test_hash_chain():
    s=generate_scenario('x','ignored_warning',1)
    assert verify_chain(s.events)


def test_complete_evidence_identifies_key_cases_with_transparent_trace_r():
    for fam in ['ignored_warning','human_override','model_defect','policy_defect','cloud_defect']:
        s=generate_scenario('x',fam,1)
        r=reconstruct(s,s.events,'trace-r')
        assert r['prediction']==s.responsible


def test_missing_critical_channel_abstains():
    s=generate_scenario('x','unauthorized_delegation',1)
    ev=[e for e in s.events if e.channel!='delegation']
    r=reconstruct(s,ev,'trace-r')
    assert r['prediction']=='NOT_IDENTIFIABLE'
    assert r['effective_observability']==0.0


def test_witness_hardening_preserves_claim_after_one_primary_channel_loss():
    s=generate_scenario('x','unauthorized_delegation',1,hardened=True)
    ev=[e for e in s.events if e.channel!='delegation']
    r=reconstruct(s,ev,'trace-r')
    assert 'delegation' not in r['missing_channels']


def test_two_channel_cut_breaks_hardened_claim():
    s=generate_scenario('x','unauthorized_delegation',1,hardened=True)
    ev=[e for e in s.events if e.channel not in {'delegation','receipt'}]
    r=reconstruct(s,ev,'trace-r')
    assert r['prediction']=='NOT_IDENTIFIABLE'


def test_fano_bound_and_cost_are_well_formed():
    assert 0 <= fano_error_lower_bound(0.5,6) <= 1
    c=channel_cost(['identity','authorization'])
    assert c['privacy']>0 and c['scalar']>0


def test_drep_separates_source_and_inference():
    from trace_r.evidence import build_drep
    s=generate_scenario('x','human_override',2)
    r=reconstruct(s,s.events,'trace-r')
    d=build_drep(s,s.events,r)
    assert 'source_evidence' in d and 'inferred_propositions' in d
    assert d['inferred_propositions']['prediction']==s.responsible


def test_responsibility_code_erasure_guarantee():
    from trace_r.coding import repetition_codebook, summarize_codebook, erasure_guarantee_holds
    cb=repetition_codebook(['a','b','c'],2)
    s=summarize_codebook(cb)
    assert s.distance == 4
    assert s.erasures_correctable == 3
    assert erasure_guarantee_holds(cb,3)
    assert not erasure_guarantee_holds(cb,4)


def test_ed25519_and_replay_guard():
    from trace_r.crypto import SigningIdentity, signed_cross_domain_receipt, verify_signed_record, ReplayGuard
    ident=SigningIdentity.generate('A')
    rec=signed_cross_domain_receipt('abc','authorization',ident,'B',1,timestamp_ns=123)
    assert verify_signed_record(rec)
    bad=dict(rec); bad['claim']='permission'
    assert not verify_signed_record(bad)
    g=ReplayGuard(); assert g.accept(rec); assert not g.accept(rec)


def test_mixed_error_erasure_condition_and_optimizer():
    from trace_r.coding import mixed_error_erasure_guarantee, optimize_responsibility_code
    assert mixed_error_erasure_guarantee(5,1,2)  # 2*1+2 < 5
    assert not mixed_error_erasure_guarantee(4,1,2)
    ans=optimize_responsibility_code(['a','b','c','d'],erasures=1,corruptions=1)
    assert ans['success'] and ans['distance'] >= 4 and ans['guarantee']


def test_js_and_bhattacharyya_bounds():
    import numpy as np
    from trace_r.theory import jensen_shannon_divergence, bhattacharyya_multiclass_upper_bound, exact_bayes_error_discrete
    assert abs(jensen_shannon_divergence([1,0],[1,0])) < 1e-12
    assert 0 <= jensen_shannon_divergence([1,0],[0,1]) <= 1
    cond=np.array([[1,0],[0,1]],float)
    assert bhattacharyya_multiclass_upper_bound(cond,[.5,.5]) == 0.0
    joint=np.array([[.5,0],[0,.5]])
    assert exact_bayes_error_discrete(joint) == 0.0
