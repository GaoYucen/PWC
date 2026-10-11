#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import heapq
import io
import json
import math
import multiprocessing as mp
import pickle
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import networkx as nx
import numpy as np
import osmnx as ox

from scripts.public_benchmark.porto_build import (
    RAW_BBOX, SAMPLE_SECONDS, build_edge_index, build_weights, haversine_m,
    highway_name, hour_of_week, polyline_length_m, speed_prior,
    to_simple_digraph, write_csv_gz, write_jsonl_gz,
)

PORTO_TZ = ZoneInfo("Europe/Lisbon")
_DG = None
_EDGE_IDX = None
_LENGTH = None

def _init_worker(dg, edge_idx, length_arr):
    global _DG, _EDGE_IDX, _LENGTH
    _DG, _EDGE_IDX, _LENGTH = dg, edge_idx, length_arr

def _bounded_path(u, v, gps_distance_m):
    if u == v:
        return [u], 0.0
    cutoff=max(700.0,min(6000.0,gps_distance_m*4.0+500.0))
    try:
        dist,path=nx.single_source_dijkstra(_DG,u,target=v,cutoff=cutoff,weight="length")
        return path,float(dist)
    except (nx.NetworkXNoPath,nx.NodeNotFound):
        return None,None

def _map_presnapped(item):
    pts=item["points"]; nn=item["nn"]; snap=np.asarray(item["snap"],dtype=float)
    if float(np.quantile(snap,.95))>100.0 or float(np.max(snap))>180.0:
        return None,"snap_distance"
    route_nodes=[]; route_edges=[]; segs=[]; total=0.0
    for i,(a,b) in enumerate(zip(pts[:-1],pts[1:])):
        u,v=int(nn[i]),int(nn[i+1])
        gps_d=haversine_m(a[0],a[1],b[0],b[1])
        if u==v: continue
        path,bridge=_bounded_path(u,v,gps_d)
        if path is None or len(path)<2: return None,"no_bridge"
        if gps_d>=30.0 and bridge/gps_d>4.5: return None,"segment_stretch"
        eids=[]; lens=[]
        for x,y in zip(path[:-1],path[1:]):
            eid=_EDGE_IDX.get((int(x),int(y)))
            if eid is None: return None,"edge_index"
            eids.append(eid); lens.append(float(_LENGTH[eid]))
        if not eids: continue
        sl=float(sum(lens))
        alloc=[SAMPLE_SECONDS*l/sl for l in lens]
        segs.append((hour_of_week(item["timestamp"]+i*SAMPLE_SECONDS),eids,alloc))
        total+=sl
        if not route_nodes: route_nodes.extend(int(x) for x in path)
        else: route_nodes.extend(int(x) for x in path[1:])
        route_edges.extend(eids)
    if len(route_edges)<3: return None,"too_short_matched"
    ratio=total/max(float(item["gps_length_m"]),1.0)
    if not (.60<=ratio<=2.60): return None,"trip_stretch"
    return {
      "trip_id":hashlib.sha256(item["trip_id"].encode()).hexdigest()[:20],
      "timestamp":int(item["timestamp"]),
      "start_hour_of_week":hour_of_week(item["timestamp"]),
      "gps_point_count":len(pts),
      "gps_length_m":float(item["gps_length_m"]),
      "matched_length_m":float(total),
      "stretch_ratio":float(ratio),
      "snap_p50_m":float(np.median(snap)),
      "snap_p95_m":float(np.quantile(snap,.95)),
      "route_nodes":route_nodes,
      "route_edges":route_edges,
      "_segments":segs,
    },None

def parse_local(s):
    dt=datetime.fromisoformat(s)
    if dt.tzinfo is None: dt=dt.replace(tzinfo=PORTO_TZ)
    return dt.astimezone(PORTO_TZ)

def scan_hourly_reservoir(raw_zip,start,end,cap,seed):
    heaps=defaultdict(list); stats=Counter(); sb=str(seed).encode()
    with zipfile.ZipFile(raw_zip) as z:
        names=[n for n in z.namelist() if n.lower().endswith(".csv")]
        with z.open(names[0]) as raw:
            r=csv.DictReader(io.TextIOWrapper(raw,encoding="utf-8",errors="replace",newline=""))
            for row in r:
                stats["rows"]+=1
                try: ts=int(row["TIMESTAMP"])
                except: continue
                dt=datetime.fromtimestamp(ts,tz=timezone.utc).astimezone(PORTO_TZ)
                if not (start<=dt<end): continue
                if str(row.get("MISSING_DATA","")).strip().lower() not in {"false","0"}:
                    stats["missing"]+=1; continue
                try: pts=json.loads(row["POLYLINE"])
                except: stats["parse_error"]+=1; continue
                if not (8<=len(pts)<=160): stats["point_gate"]+=1; continue
                try: pts=[(float(p[0]),float(p[1])) for p in pts]
                except: stats["coord_error"]+=1; continue
                if not all(RAW_BBOX[0]<=lat<=RAW_BBOX[1] and RAW_BBOX[2]<=lon<=RAW_BBOX[3] for lon,lat in pts):
                    stats["bbox_gate"]+=1; continue
                gl=polyline_length_m(pts)
                if not (500<=gl<=35000): stats["length_gate"]+=1; continue
                mx=max(haversine_m(a[0],a[1],b[0],b[1]) for a,b in zip(pts[:-1],pts[1:]))
                if mx>2500: stats["jump_gate"]+=1; continue
                hour=dt.replace(minute=0,second=0,microsecond=0)
                tid=str(row["TRIP_ID"])
                h=int.from_bytes(hashlib.blake2b(sb+tid.encode(),digest_size=8).digest(),"big")
                item={"hash":h,"trip_id":tid,"timestamp":ts,"points":pts,"gps_length_m":gl,
                      "hour_iso":hour.isoformat()}
                heap=heaps[hour]
                if len(heap)<cap: heapq.heappush(heap,(-h,item))
                elif h < -heap[0][0]: heapq.heapreplace(heap,(-h,item))
                stats["eligible"]+=1
    out={}
    for hour,heap in heaps.items():
        xs=[x[1] for x in heap]; xs.sort(key=lambda q:q["hash"]); out[hour]=xs
    return out,stats

def batch_snap(g,items,chunk_trips=1000):
    for a in range(0,len(items),chunk_trips):
        chunk=items[a:a+chunk_trips]
        lens=[len(x["points"]) for x in chunk]
        X=[];Y=[]
        for x in chunk:
            X.extend(p[0] for p in x["points"]); Y.extend(p[1] for p in x["points"])
        nn,dist=ox.distance.nearest_nodes(g,X=np.asarray(X),Y=np.asarray(Y),return_dist=True)
        nn=np.asarray(nn); dist=np.asarray(dist)
        pos=0
        for x,n in zip(chunk,lens):
            x["nn"]=[int(v) for v in nn[pos:pos+n]]
            x["snap"]=[float(v) for v in dist[pos:pos+n]]
            pos+=n

def map_split(label,g,dg,edge_idx,length_arr,hours,workers):
    flat=[]; hour_order=sorted(hours)
    for h in hour_order: flat.extend(hours[h])
    print(f"{label}: raw selected={len(flat):,}; snapping",flush=True)
    batch_snap(g,flat)
    print(f"{label}: mapping with {workers} workers",flush=True)
    ctx=mp.get_context("fork")
    with ctx.Pool(workers,initializer=_init_worker,initargs=(dg,edge_idx,length_arr)) as pool:
        results=list(pool.imap(_map_presnapped,flat,chunksize=8))
    matched_by_hour=defaultdict(list); reject=Counter()
    for item,(trip,why) in zip(flat,results):
        h=datetime.fromisoformat(item["hour_iso"])
        if trip is None: reject[why]+=1
        else: matched_by_hour[h].append(trip)
    for h in matched_by_hour: matched_by_hour[h].sort(key=lambda x:x["timestamp"])
    counts=[len(matched_by_hour.get(h,[])) for h in hour_order]
    print(f"{label}: accepted={sum(counts):,}, min/hour={min(counts) if counts else 0}, median/hour={np.median(counts) if counts else 0}",flush=True)
    return matched_by_hour,reject

def dense_topology(out,dg,pairs,length_arr):
    node_ids=np.unique(np.array([x for p in pairs for x in p],dtype=np.int64))
    node_to_idx={int(x):i for i,x in enumerate(node_ids)}
    edge_u=np.asarray([node_to_idx[u] for u,v in pairs],dtype=np.int32)
    edge_v=np.asarray([node_to_idx[v] for u,v in pairs],dtype=np.int32)
    n_nodes=len(node_ids)
    keys=edge_u.astype(np.int64)*n_nodes+edge_v.astype(np.int64)
    sort=np.argsort(keys,kind="stable").astype(np.int32)
    sk=keys[sort]
    starts=np.r_[0,1+np.flatnonzero(sk[1:]!=sk[:-1])].astype(np.int32)
    uk=sk[starts]; uu=(uk//n_nodes).astype(np.int32); uv=(uk%n_nodes).astype(np.int32)
    indptr=np.r_[0,np.cumsum(np.bincount(uu,minlength=n_nodes),dtype=np.int64)]
    np.savez(out/"topology.npz",link_ids=np.arange(len(pairs),dtype=np.int64),node_ids=node_ids,
             edge_u=edge_u,edge_v=edge_v,sort_order=sort,group_starts=starts,indices=uv,
             indptr=indptr,n_nodes=np.asarray([n_nodes],dtype=np.int64))
    return node_to_idx

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",required=True)
    ap.add_argument("--raw-zip",required=True)
    ap.add_argument("--test-week-start",required=True,help="Monday 00:00 local, ISO date/time")
    ap.add_argument("--train-weeks",type=int,default=4)
    ap.add_argument("--val-weeks",type=int,default=1)
    ap.add_argument("--raw-cap-per-hour",type=int,default=120)
    ap.add_argument("--workers",type=int,default=8)
    ap.add_argument("--seed",type=int,default=20261011)
    ap.add_argument("--name",default="porto-medium-hourly-v0.1")
    args=ap.parse_args()
    root=Path(args.root); out=root/args.name; out.mkdir(parents=True,exist_ok=True); (out/"weights").mkdir(exist_ok=True)
    test_start=parse_local(args.test_week_start)
    val_start=test_start-timedelta(weeks=args.val_weeks)
    train_start=val_start-timedelta(weeks=args.train_weeks)
    end=test_start+timedelta(weeks=1)
    print(f"range train={train_start} val={val_start} test={test_start} end={end}",flush=True)

    g=ox.io.load_graphml(root/"porto_drive.graphml"); dg=to_simple_digraph(g)
    pairs,edge_idx,length_arr,highways,edge_rows=build_edge_index(dg)
    hours,scan=scan_hourly_reservoir(Path(args.raw_zip),train_start,end,args.raw_cap_per_hour,args.seed)
    def slice_hours(a,b):
        result={}
        h=a
        while h<b:
            result[h]=hours.get(h,[])
            h+=timedelta(hours=1)
        return result
    train_h=slice_hours(train_start,val_start)
    val_h=slice_hours(val_start,test_start)
    test_h=slice_hours(test_start,end)

    train_m,rj_train=map_split("train",g,dg,edge_idx,length_arr,train_h,args.workers)
    val_m,rj_val=map_split("val",g,dg,edge_idx,length_arr,val_h,args.workers)
    test_m,rj_test=map_split("test",g,dg,edge_idx,length_arr,test_h,args.workers)

    train=[x for h in sorted(train_m) for x in train_m[h]]
    val=[x for h in sorted(val_m) for x in val_m[h]]
    test=[x for h in sorted(test_m) for x in test_m[h]]
    weights=build_weights(train,length_arr,highways,len(pairs))
    for k,v in weights.items():
        if not np.all(np.isfinite(v)): raise RuntimeError(f"nonfinite {k}")

    write_csv_gz(out/"edges.csv.gz",edge_rows,["edge_idx","u","v","length_m","highway","name","osmid"])
    node_rows=[{"node_id":int(n),"lon":float(d.get("x",0)),"lat":float(d.get("y",0))} for n,d in dg.nodes(data=True)]
    write_csv_gz(out/"nodes.csv.gz",node_rows,["node_id","lon","lat"])
    write_jsonl_gz(out/"trajectories_train.jsonl.gz",train)
    write_jsonl_gz(out/"trajectories_val.jsonl.gz",val)
    write_jsonl_gz(out/"trajectories_test.jsonl.gz",test)
    for name,arr in weights.items(): np.save(out/"weights"/f"{name}.npy",arr)

    node_to_idx=dense_topology(out,dg,pairs,length_arr)
    np.save(out/"weight_length.npy",weights["length"])
    np.save(out/"weight_time.npy",weights["travel_time_static"])
    np.save(out/"weight_power_s.npy",weights["preference_static"])
    np.save(out/"weight_power_d.npy",weights["preference_dynamic_168"])

    def make_slots(start,mapped):
        slots=[]
        counts=[]
        for i in range(168):
            h=start+timedelta(hours=i)
            xs=mapped.get(h,[])
            orders=[]
            for tr in xs:
                rn=tuple(node_to_idx[int(x)] for x in tr["route_nodes"])
                orders.append({"order_id":tr["trip_id"],"timestamp":tr["timestamp"],"aux":"porto-hourly",
                               "start_node":rn[0],"end_node":rn[-1],"route_nodes":rn})
            slots.append(orders); counts.append(len(orders))
        return slots,counts
    val_slots,val_counts=make_slots(val_start,val_m)
    test_slots,test_counts=make_slots(test_start,test_m)
    with (out/"validation_orders.pkl").open("wb") as f: pickle.dump(val_slots,f,pickle.HIGHEST_PROTOCOL)
    with (out/"sampled_orders.pkl").open("wb") as f: pickle.dump(test_slots,f,pickle.HIGHEST_PROTOCOL)

    meta={
      "name":args.name,
      "train_start":train_start.isoformat(),"validation_start":val_start.isoformat(),
      "test_start":test_start.isoformat(),"end":end.isoformat(),
      "train_weeks":args.train_weeks,"validation_weeks":args.val_weeks,
      "raw_cap_per_hour":args.raw_cap_per_hour,
      "mapped":{"train":len(train),"validation":len(val),"test":len(test)},
      "validation_per_hour":val_counts,"test_per_hour":test_counts,
      "validation_min_per_hour":min(val_counts),"test_min_per_hour":min(test_counts),
      "validation_hours_ge_5":sum(x>=5 for x in val_counts),"test_hours_ge_5":sum(x>=5 for x in test_counts),
      "validation_hours_ge_10":sum(x>=10 for x in val_counts),"test_hours_ge_10":sum(x>=10 for x in test_counts),
      "train_observed_edges":int(np.count_nonzero(weights["train_edge_traversal_count"]>0)),
      "scan_stats":dict(scan),
      "rejects":{"train":dict(rj_train),"validation":dict(rj_val),"test":dict(rj_test)},
      "leakage_rule":"all edge weights from training weeks only; validation selects hyperparameters; test is untouched",
    }
    (out/"metadata.json").write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2),flush=True)

if __name__=="__main__": main()
