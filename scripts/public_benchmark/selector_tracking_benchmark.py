#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math, pickle
from collections import deque
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pwc.experiment import WeightBank
from pwc.router import RoadRouter

MODELS = ("Length","TravelTime-Static","Preference-Static","Preference-Dynamic")


def softmax(x):
    x=np.asarray(x,dtype=float)
    x=x-np.max(x)
    e=np.exp(x)
    return e/e.sum()


class Hedge:
    def __init__(self,k,eta):
        self.eta=eta; self.score=np.zeros(k)
    def probs(self): return softmax(self.eta*self.score)
    def update(self,r): self.score += r


class FixedShare:
    def __init__(self,k,eta,alpha):
        self.k=k; self.eta=eta; self.alpha=alpha
        self.p=np.ones(k)/k
    def probs(self): return self.p.copy()
    def update(self,r):
        v=self.p*np.exp(self.eta*np.asarray(r))
        v=v/v.sum()
        if self.k==1:
            self.p=v
        else:
            self.p=(1-self.alpha)*v + self.alpha*(1-v)/(self.k-1)
        self.p=self.p/self.p.sum()


class DiscountedHedge:
    def __init__(self,k,eta,gamma):
        self.eta=eta; self.gamma=gamma; self.score=np.zeros(k)
    def probs(self): return softmax(self.eta*self.score)
    def update(self,r): self.score=self.gamma*self.score+np.asarray(r)


class SlidingHedge:
    def __init__(self,k,eta,window):
        self.eta=eta; self.window=window; self.hist=deque(maxlen=window); self.k=k
    def probs(self):
        if not self.hist: score=np.zeros(self.k)
        else: score=np.sum(np.asarray(self.hist),axis=0)
        return softmax(self.eta*score)
    def update(self,r): self.hist.append(np.asarray(r,dtype=float))


class FPL:
    def __init__(self,k,eta,seed):
        self.eta=eta; self.score=np.zeros(k); self.rng=np.random.default_rng(seed)
    def choose(self):
        z=self.rng.exponential(scale=1.0/self.eta,size=len(self.score))
        return int(np.argmax(self.score+z))
    def update(self,r): self.score += r


def expected_reward(factory,R):
    s=factory()
    vals=[]
    for r in R:
        p=s.probs()
        vals.append(float(np.dot(p,r)))
        s.update(r)
    return float(np.mean(vals)), vals


def sampled_sequence(factory,R,seed):
    s=factory(); rng=np.random.default_rng(seed); seq=[]; direct=[]
    for r in R:
        p=s.probs()
        m=int(rng.choice(len(p),p=p))
        seq.append(m); direct.append(float(r[m]))
        s.update(r)
    return seq,float(np.mean(direct))


def fpl_sequence(R,eta,seed):
    s=FPL(R.shape[1],eta,seed); seq=[]; vals=[]
    for r in R:
        m=s.choose(); seq.append(m); vals.append(float(r[m])); s.update(r)
    return seq,float(np.mean(vals))


def pwc_weights(bank, selected, global_slots, t, L):
    first=max(0,t-L)
    out=np.zeros(bank.n_edges,dtype=np.float32)
    for s in range(first,t+1):
        m=int(selected[s]); g=int(global_slots[s])
        out += np.float32(math.exp(-(t-s))) * bank.raw(m,g)
    return out


def chase_reward(city,R_all,global_slots,seq,L):
    router=RoadRouter(city)
    bank=WeightBank(city,calibration="median")
    with (city/"sampled_orders.pkl").open("rb") as f: orders=pickle.load(f)
    vals=[]
    for t,g in enumerate(global_slots):
        w=pwc_weights(bank,seq,global_slots,t,L)
        res=router.evaluate(w,orders[g])
        vals.append(float(np.mean(res.per_order_overlap)))
    return float(np.mean(vals)),vals


def tune_prob(name,Rval):
    K=Rval.shape[1]
    eta_grid=[0.5,1.0,2.0,3.0,5.0,10.0,20.0,50.0]
    candidates=[]
    if name=="Hedge":
        for eta in eta_grid:
            factory=lambda eta=eta:Hedge(K,eta)
            score,_=expected_reward(factory,Rval)
            candidates.append((score,{"eta":eta},factory))
    elif name=="FixedShare":
        for eta in eta_grid:
            for alpha in [0.01,0.02,0.05,0.1,0.2,0.3,0.5]:
                factory=lambda eta=eta,alpha=alpha:FixedShare(K,eta,alpha)
                score,_=expected_reward(factory,Rval)
                candidates.append((score,{"eta":eta,"alpha":alpha},factory))
    elif name=="DiscountedHedge":
        for eta in eta_grid:
            for gamma in [0.0,0.25,0.5,0.7,0.8,0.9,0.95]:
                factory=lambda eta=eta,gamma=gamma:DiscountedHedge(K,eta,gamma)
                score,_=expected_reward(factory,Rval)
                candidates.append((score,{"eta":eta,"gamma":gamma},factory))
    elif name=="SlidingHedge":
        for eta in [1.0,3.0,10.0,30.0]:
            for window in [1,2,3,5,8,12,20]:
                factory=lambda eta=eta,window=window:SlidingHedge(K,eta,window)
                score,_=expected_reward(factory,Rval)
                candidates.append((score,{"eta":eta,"window":window},factory))
    candidates.sort(key=lambda z:z[0],reverse=True)
    return candidates[0],[(s,p) for s,p,_ in candidates[:5]]


def tune_fpl(Rval):
    rows=[]
    for eta in [0.3,0.5,1.0,2.0,3.0,5.0,10.0,20.0,50.0]:
        vals=[]
        for seed in range(30):
            _,v=fpl_sequence(Rval,eta,seed)
            vals.append(v)
        rows.append((float(np.mean(vals)),float(np.std(vals)),eta))
    rows.sort(reverse=True)
    return rows[0],rows[:5]


def run_one(root,n,L,seeds):
    metrics=json.loads((root/f"N{n}"/"metrics.json").read_text())
    R=np.asarray(metrics["slot_baseline_overlap"],dtype=float)
    city=Path(str(root).replace("pwc-smoke-continuous-v0.1","pwc-smoke-city-v0.1"))/f"N{n}"
    T=len(R)
    val_trips=142
    val_slots=val_trips//n
    boundary=val_slots
    test_start=boundary+1
    val_idx=np.arange(0,val_slots,dtype=int)
    test_idx=np.arange(test_start,T,dtype=int)
    Rval=R[val_idx]
    Rtest=R[test_idx]
    best_fixed=float(np.max(Rtest.mean(axis=0)))
    best_idx=int(np.argmax(Rtest.mean(axis=0)))
    oracle=float(np.mean(np.max(Rtest,axis=1)))

    result={
      "N":n,"T_all":T,"T_val":len(val_idx),"T_test":len(test_idx),
      "val_indices":[int(x) for x in val_idx],
      "test_indices":[int(x) for x in test_idx],
      "best_fixed_model":MODELS[best_idx],
      "best_fixed_test_pct":100*best_fixed,
      "full_oracle_test_pct":100*oracle,
      "oracle_headroom_pp":100*(oracle-best_fixed),
      "algorithms":[]
    }

    # Original FPL beta=3 and validation-tuned FPL.
    tuned_fpl, top_fpl=tune_fpl(Rval)
    for label,eta in [("FPL-beta3",3.0),("FPL-val-tuned",float(tuned_fpl[2]))]:
        direct=[]; chase=[]
        for seed in range(seeds):
            seq,d=fpl_sequence(Rtest,eta,seed)
            c,_=chase_reward(city,R,test_idx,seq,L)
            direct.append(d); chase.append(c)
        result["algorithms"].append({
          "name":label,"params":{"eta":eta},
          "validation_top5":top_fpl if label=="FPL-val-tuned" else None,
          "test_direct_mean_pct":100*float(np.mean(direct)),
          "test_direct_std_pp":100*float(np.std(direct)),
          "test_chase_mean_pct":100*float(np.mean(chase)),
          "test_chase_std_pp":100*float(np.std(chase)),
          "direct_gap_to_oracle_pp":100*(oracle-float(np.mean(direct))),
          "chase_gap_to_oracle_pp":100*(oracle-float(np.mean(chase))),
          "chase_minus_best_fixed_pp":100*(float(np.mean(chase))-best_fixed),
        })

    for name in ["Hedge","FixedShare","DiscountedHedge","SlidingHedge"]:
        (vscore,params,factory),top=tune_prob(name,Rval)
        expected,_=expected_reward(factory,Rtest)
        sampled_direct=[]; chase=[]
        for seed in range(seeds):
            seq,d=sampled_sequence(factory,Rtest,1000+seed)
            c,_=chase_reward(city,R,test_idx,seq,L)
            sampled_direct.append(d); chase.append(c)
        result["algorithms"].append({
          "name":name,"params":params,
          "validation_expected_direct_pct":100*vscore,
          "validation_top5":[{"score_pct":100*s,"params":p} for s,p in top],
          "test_expected_direct_pct":100*expected,
          "test_sampled_direct_mean_pct":100*float(np.mean(sampled_direct)),
          "test_sampled_direct_std_pp":100*float(np.std(sampled_direct)),
          "test_chase_mean_pct":100*float(np.mean(chase)),
          "test_chase_std_pp":100*float(np.std(chase)),
          "direct_gap_to_oracle_pp":100*(oracle-expected),
          "chase_gap_to_oracle_pp":100*(oracle-float(np.mean(chase))),
          "chase_minus_best_fixed_pp":100*(float(np.mean(chase))-best_fixed),
        })
    result["algorithms"].sort(key=lambda x:x["test_chase_mean_pct"],reverse=True)
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--seeds",type=int,default=20)
    ap.add_argument("--L",type=int,default=3)
    args=ap.parse_args()
    root=Path(args.root); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    runs=[run_one(root,n,args.L,args.seeds) for n in [5,10,20]]
    summary={"protocol":{
      "tuning":"hyperparameters selected only on pure validation event slots",
      "test":"pure test event slots; boundary mixed slot discarded",
      "feedback":"full-information base rewards revealed after each slot",
      "chasing_L":args.L,
      "stochastic_seeds":args.seeds,
    },"runs":runs}
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
