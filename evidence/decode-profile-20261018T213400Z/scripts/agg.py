import gzip, json, sys, collections, glob
def load(f):
    d=json.load(gzip.open(f))
    ev=[e for e in d["traceEvents"] if e.get("ph")=="X" and e.get("cat") in ("kernel","gpu_memcpy","gpu_memset")]
    return ev
for tag in ("c1","c16"):
    f=sorted(glob.glob(f"/tmp/opencode/prof/{tag}/*TP-0-*"))[0]
    ev=load(f)
    t0=min(e["ts"] for e in ev); t1=max(e["ts"]+e["dur"] for e in ev)
    tot=sum(e["dur"] for e in ev)
    agg=collections.defaultdict(lambda:[0,0])
    for e in ev:
        n=e["name"]; agg[n][0]+=e["dur"]; agg[n][1]+=1
    print(f"=== {tag}: span {(t1-t0)/1e3:.1f} ms, kernel-busy {tot/1e3:.1f} ms, kernels {len(ev)}")
    for n,(d,c) in sorted(agg.items(),key=lambda x:-x[1][0])[:22]:
        print(f"{d/1e3:9.2f} ms {100*d/tot:5.1f}% n={c:6d}  {n[:150]}")
