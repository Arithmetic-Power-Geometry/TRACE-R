from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple
import base64, hashlib, json, os, statistics, time, uuid
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives import serialization


def canonical(obj: Dict) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('utf-8')


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class SigningIdentity:
    name: str
    private_key: Ed25519PrivateKey

    @classmethod
    def generate(cls, name: str) -> 'SigningIdentity':
        return cls(name, Ed25519PrivateKey.generate())

    def public_b64(self) -> str:
        raw = self.private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw)
        return base64.b64encode(raw).decode()

    def sign(self, payload: Dict) -> Dict:
        body = dict(payload)
        body.setdefault('nonce', str(uuid.uuid4()))
        body.setdefault('signer', self.name)
        msg = canonical(body)
        sig = self.private_key.sign(msg)
        return {**body, 'signature': base64.b64encode(sig).decode(), 'public_key': self.public_b64()}


def verify_signed_record(record: Dict) -> bool:
    try:
        body = {k: v for k, v in record.items() if k not in {'signature', 'public_key'}}
        pk = Ed25519PublicKey.from_public_bytes(base64.b64decode(record['public_key']))
        pk.verify(base64.b64decode(record['signature']), canonical(body))
        return True
    except Exception:
        return False


class ReplayGuard:
    def __init__(self): self._seen=set()
    def accept(self, record: Dict) -> bool:
        nonce=record.get('nonce')
        if not nonce or nonce in self._seen: return False
        if not verify_signed_record(record): return False
        self._seen.add(nonce); return True


def merkle_root(records: Iterable[Dict]) -> str:
    leaves=[hashlib.sha256(canonical(r)).digest() for r in records]
    if not leaves: return sha256_hex(b'')
    while len(leaves) > 1:
        if len(leaves) % 2: leaves.append(leaves[-1])
        leaves=[hashlib.sha256(leaves[i]+leaves[i+1]).digest() for i in range(0,len(leaves),2)]
    return leaves[0].hex()


def signed_cross_domain_receipt(event_digest: str, claim: str, identity: SigningIdentity,
                                counterparty: str, seq: int, timestamp_ns: int|None=None) -> Dict:
    payload={'event_digest':event_digest,'claim':claim,'counterparty':counterparty,
             'seq':int(seq),'timestamp_ns':int(timestamp_ns or time.time_ns())}
    return identity.sign(payload)


def benchmark_crypto(n: int=3000, payload_bytes: int=256) -> Dict[str,float]:
    identity=SigningIdentity.generate('benchmark-agent')
    payload={'seq':0,'blob':'x'*payload_bytes,'timestamp_ns':time.time_ns()}
    sign_ms=[]; verify_ms=[]; sizes=[]; records=[]
    for i in range(n):
        payload['seq']=i; payload['nonce']=str(i)
        t=time.perf_counter_ns(); rec=identity.sign(payload); sign_ms.append((time.perf_counter_ns()-t)/1e6)
        t=time.perf_counter_ns(); ok=verify_signed_record(rec); verify_ms.append((time.perf_counter_ns()-t)/1e6)
        if not ok: raise RuntimeError('signature verification failed')
        sizes.append(len(canonical(rec))); records.append(rec)
    t=time.perf_counter_ns(); root=merkle_root(records); merkle_ms=(time.perf_counter_ns()-t)/1e6
    return {
        'n':n,'payload_bytes':payload_bytes,
        'sign_median_ms':statistics.median(sign_ms),
        'sign_p95_ms':sorted(sign_ms)[int(.95*(len(sign_ms)-1))],
        'verify_median_ms':statistics.median(verify_ms),
        'verify_p95_ms':sorted(verify_ms)[int(.95*(len(verify_ms)-1))],
        'record_bytes_mean':statistics.mean(sizes),
        'verify_per_second':1000.0/max(statistics.mean(verify_ms),1e-12),
        'merkle_ms_total':merkle_ms,'merkle_root':root,
    }
