#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, json, math, pickle, platform, statistics, sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pwc.experiment import WeightBank
from pwc.router import RoadRouter

MODELS=("Length","TravelTime-Static","Preference-Static","Preference-Dynamic")
ETA_GRID=(0.5,1.0,2.0,3.0,5.0,10.0,20.0,50.0)

def softmax(x):
    x=np.asarray(x,float); x=x-x.max()
    e=np.exp(x)
    return e/e.sum()

class Hedge:
    def __init__(self,k,eta): self.s=np.zeros(k); self.eta=eta
    def probs(self): return softmax(self.eta*self.s)
    def update(self,r): self.s+=np.asarray(r)

class FixedShare:
    def __init__(self,k,eta,alpha):
        self.k=k; self.eta=eta; self.alpha=alpha; self.p=np.ones(k)/k
    def probs(self): return self.p.copy()
    def update(self,r):
        v=self.p*np.exp(self.eta*np.asarray(r)); v/=v.sum()
        self.p=(1-self.alpha)*v+self.alpha*(1-v)/(self.k-1)
        self.p/=self.p.sum()

class DiscountedHedge:
    def __init__(self,k,eta,gamma): self.s=np.zeros(k); self.eta=eta; self.gamma=gamma
    def probs(self): return softmax(self.eta*self.s)
    def update(self,r): self.s=self.gamma*self.s+np.asarray(r)

class SlidingHedge:
    def __init__(self,k,eta,window): self.k=k; self.eta=eta; self.h=deque(maxlen=window)
    def probs(self):
        s=np.zeros(self.k) if not self.h else np.sum(np.asarray(self.h),axis=0)
        return softmax(self.eta*s)
    def update(self,r): self.h.append(np.asarray(r,float))

class FPL:
    def __init__(self,k,eta,seed):
        self.s=np.zeros(k); self.eta=eta; self.rng=np.random.default_rng(seed)
    def choose(self):
        z=self.rng.exponential(scale=1.0/self.eta,size=len(self.s))
        return int(np.argmax(self.s+z))
    def update(self,r): self.s+=np.asarray(r)

def load_orders(path):
    with Path(path).open("rb") as f: return pickle.load(f)

def baseline_rewards(city,orders,n_orders=None):
    router=RoadRouter(city); bank=WeightBank(city,calibration="median")
    T=len(orders); R=np.zeros((T,len(MODELS)),float); counts=[]
    for t in range(T):
        slot=orders[t] if n_orders is None else orders[t][:n_orders]
        if not slot: raise RuntimeError(f"empty slot {t}")
        counts.append(len(slot))
        for m in range(len(MODELS)):
            res=router.evaluate(bank.base_effective(m,t,3),slot)
            R[t,m]=float(np.mean(res.per_order_overlap))
        if (t+1)%24==0: print(f"baseline {t+1}/{T}",flush=True)
    return R,counts

def expected_reward(factory,R):
    s=factory(); vals=[]
    for r in R:
        p=s.probs(); vals.append(float(p@r)); s.update(r)
    return float(np.mean(vals))

def sampled_sequence(factory,R,seed):
    s=factory(); rng=np.random.default_rng(seed); seq=[]; vals=[]
    for r in R:
        p=s.probs(); m=int(rng.choice(len(p),p=p))
        seq.append(m); vals.append(float(r[m])); s.update(r)
    return seq,float(np.mean(vals))

def fpl_sequence(R,eta,seed):
    s=FPL(R.shape[1],eta,seed); seq=[]; vals=[]
    for r in R:
        m=s.choose(); seq.append(m); vals.append(float(r[m])); s.update(r)
    return seq,float(np.mean(vals))

def tune_prob(kind,R):
    k=R.shape[1]; rows=[]
    if kind=="Hedge":
        for eta in ETA_GRID:
            f=lambda eta=eta:Hedge(k,eta); rows.append((expected_reward(f,R),{"eta":eta},f))
    elif kind=="FixedShare":
        for eta in ETA_GRID:
            for alpha in (0.005,0.01,0.02,0.05,0.1,0.2,0.3,0.5):
                f=lambda eta=eta,alpha=alpha:FixedShare(k,eta,alpha)
                rows.append((expected_reward(f,R),{"eta":eta,"alpha":alpha},f))
    elif kind=="DiscountedHedge":
        for eta in ETA_GRID:
            for gamma in (0.0,0.25,0.5,0.7,0.8,0.9,0.95,0.98):
                f=lambda eta=eta,gamma=gamma:DiscountedHedge(k,eta,gamma)
                rows.append((expected_reward(f,R),{"eta":eta,"gamma":gamma},f))
    elif kind=="SlidingHedge":
        for eta in (1.0,3.0,10.0,30.0,50.0):
            for window in (1,2,3,5,8,12,24,48):
                f=lambda eta=eta,window=window:SlidingHedge(k,eta,window)
                rows.append((expected_reward(f,R),{"eta":eta,"window":window},f))
    rows.sort(key=lambda z:z[0],reverse=True)
    return rows[0],rows[:5]

def tune_fpl(R,seeds=100):
    rows=[]
    for eta in (0.3,0.5,1,2,3,5,10,20,50,100):
        vals=[fpl_sequence(R,eta,s)[1] for s in range(seeds)]
        rows.append((float(np.mean(vals)),float(np.std(vals)),eta))
    rows.sort(reverse=True)
    return rows[0],rows[:5]

def chase_weight(bank,seq,t,L=3):
    first=max(0,t-L); out=np.zeros(bank.n_edges,dtype=np.float32)
    for s in range(first,t+1):
        out += np.float32(math.exp(-(t-s))) * bank.raw(int(seq[s]),s)
    return out

def chase_reward(city,orders,seq,n_orders=5,L=3):
    router=RoadRouter(city); bank=WeightBank(city,calibration="median"); vals=[]
    for t in range(len(orders)):
        slot=orders[t][:n_orders]
        res=router.evaluate(chase_weight(bank,seq,t,L),slot)
        vals.append(float(np.mean(res.per_order_overlap)))
    return float(np.mean(vals))

def switching_oracle(R,S):
    T,K=R.shape; neg=-1e100
    dp=np.full((S+1,K),neg); dp[0]=R[0]
    for t in range(1,T):
        new=np.full_like(dp,neg)
        for s in range(S+1):
            for m in range(K):
                best=dp[s,m]
                if s>0:
                    x=dp[s-1].copy(); x[m]=neg
                    best=max(best,float(np.max(x)))
                new[s,m]=best+R[t,m]
        dp=new
    return float(np.max(dp))/T

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--city-dir",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--N",type=int,default=5)
    ap.add_argument("--seeds",type=int,default=20)
    args=ap.parse_args()
    if args.N < 1 or args.seeds < 1:
        ap.error("N and seeds must be positive")
    city=Path(args.city_dir); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    val_orders=load_orders(city/"validation_orders.pkl")
    test_orders=load_orders(city/"sampled_orders.pkl")
    if min(len(x) for x in test_orders)<args.N:
        raise RuntimeError("test does not support requested N")
    # Validation uses all available orders per real hour to preserve all 168 hours.
    Rv,val_counts=baseline_rewards(city,val_orders,n_orders=None)
    Rt,test_counts=baseline_rewards(city,test_orders,n_orders=args.N)
    np.savez_compressed(out/"reward_matrices.npz",validation=Rv,test=Rt)

    best_idx=int(np.argmax(Rt.mean(axis=0))); best=float(Rt[:,best_idx].mean())
    oracle=float(Rt.max(axis=1).mean())
    sw={}
    for S in [0,1,2,4,8,16,24,32,48,64,96,167]:
        sw[str(S)]=100*switching_oracle(Rt,S)

    algs=[]
    seed_records=[]
    # Original and validation-tuned FPL.
    tuned,top=tune_fpl(Rv)
    for name,eta in [("FPL-beta3",3.0),("FPL-val-tuned",float(tuned[2]))]:
        direct=[]; chase=[]
        for seed in range(args.seeds):
            seq,d=fpl_sequence(Rt,eta,seed)
            c=chase_reward(city,test_orders,seq,args.N)
            direct.append(d); chase.append(c)
            seed_records.append({"name":name,"seed":seed,"selected_models":seq,
                                 "direct_pct":100*d,"chase_pct":100*c})
        algs.append({
          "name":name,"params":{"eta":eta},
          "validation_top5":top if name=="FPL-val-tuned" else None,
          "direct_pct":100*statistics.mean(direct),
          "direct_std_pp":100*statistics.pstdev(direct),
          "chase_pct":100*statistics.mean(chase),
          "chase_std_pp":100*statistics.pstdev(chase),
        })
    for kind in ("Hedge","FixedShare","DiscountedHedge","SlidingHedge"):
        (v,params,factory),top=tune_prob(kind,Rv)
        exp=expected_reward(factory,Rt); direct=[]; chase=[]
        for seed in range(args.seeds):
            seq,d=sampled_sequence(factory,Rt,1000+seed)
            c=chase_reward(city,test_orders,seq,args.N)
            direct.append(d); chase.append(c)
            seed_records.append({"name":kind,"seed":1000+seed,"selected_models":seq,
                                 "direct_pct":100*d,"chase_pct":100*c})
        algs.append({
          "name":kind,"params":params,"validation_expected_pct":100*v,
          "validation_top5":[{"pct":100*x[0],"params":x[1]} for x in top],
          "test_expected_direct_pct":100*exp,
          "direct_pct":100*statistics.mean(direct),
          "direct_std_pp":100*statistics.pstdev(direct),
          "chase_pct":100*statistics.mean(chase),
          "chase_std_pp":100*statistics.pstdev(chase),
        })
    for a in algs:
        a["chase_minus_best_fixed_pp"]=a["chase_pct"]-100*best
        a["chase_gap_to_oracle_pp"]=100*oracle-a["chase_pct"]
        a["direct_minus_best_fixed_pp"]=a["direct_pct"]-100*best
    algs.sort(key=lambda a:a["chase_pct"],reverse=True)
    summary={
      "protocol":{
        "T":168,"N_test":args.N,"L":3,
        "validation":"all available mapped orders per real hour; mean overlap reward",
        "test":"first N mapped orders per real hour; N fixed",
        "hyperparameters":"validation only; frozen on test",
        "feedback":"full-information rewards revealed after each completed hour",
        "stochastic_seeds":args.seeds,
        "FPL_seed_start":0,"probability_selector_seed_start":1000,
      },
      "validation_order_counts":{"min":min(val_counts),"median":float(np.median(val_counts)),"max":max(val_counts)},
      "test_order_counts":{"min":min(test_counts),"median":float(np.median(test_counts)),"max":max(test_counts)},
      "best_fixed_model":MODELS[best_idx],"best_fixed_pct":100*best,
      "full_oracle_pct":100*oracle,"oracle_headroom_pp":100*(oracle-best),
      "switching_oracle_pct":sw,
      "algorithms":algs,
    }
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    (out/"per_seed.json").write_text(json.dumps(seed_records,indent=2))
    def fingerprint(path):
        h=hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda:f.read(1024*1024),b""):
                h.update(block)
        return h.hexdigest()
    provenance={"python":platform.python_version(),"numpy":np.__version__,
                "script_sha256":fingerprint(Path(__file__)),
                "dataset_manifest_sha256":fingerprint(city/"MANIFEST.json") if (city/"MANIFEST.json").exists() else None,
                "protocol":summary["protocol"],
                "input_sha256":{name:fingerprint(city/name) for name in
                    ("topology.npz","sampled_orders.pkl","validation_orders.pkl",
                     "weight_length.npy","weight_time.npy","weight_power_s.npy","weight_power_d.npy")}}
    (out/"provenance.json").write_text(json.dumps(provenance,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
