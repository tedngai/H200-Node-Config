import json, time, urllib.request, threading, sys
N=int(sys.argv[1]); TAG=sys.argv[2]
BASE="http://127.0.0.1:8013"
def post(path,body,timeout=600):
    req=urllib.request.Request(BASE+path,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    return urllib.request.urlopen(req,timeout=timeout).read().decode()
def run(i):
    p=f"Write production-quality code for a concurrent job scheduler in Python (variant {i}). Include types, docstrings, error handling and unit tests."
    post("/v1/chat/completions",{"model":"dsv41-flash-1017","messages":[{"role":"user","content":p}],"max_tokens":1500})
th=[threading.Thread(target=run,args=(i,)) for i in range(N)]
[x.start() for x in th]
time.sleep(8)
m=urllib.request.urlopen(BASE+"/metrics").read().decode()
print("running_at_profile:",[l.split()[-1] for l in m.splitlines() if l.startswith("sglang:num_running_reqs")])
print(post("/start_profile",{"output_dir":f"/tmp/sglprof/{TAG}","num_steps":6,"activities":["GPU"],"profile_prefix":TAG}))
[x.join() for x in th]
print("done")
