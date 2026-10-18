import json, time, urllib.request, sys, threading, statistics as st
BASE="http://127.0.0.1:8013/v1/chat/completions"
topics=["an LRU cache with TTL in Python","a token-bucket rate limiter in Go","a trie with prefix search in Rust","a REST client with retries in TypeScript","a thread pool in C++17","a JSON schema validator in Python","a Redis-backed job queue in Node.js","a B-tree insert routine in C","an async web crawler in Python","a Kotlin coroutine-based downloader","a SQL query builder in TypeScript","a Dijkstra router in Java","a ring buffer in Rust","a markdown tokenizer in Go","a CSV streaming parser in Python","an event sourcing aggregate in C#"]
def run(i,res,mt):
    p=f"Write production-quality code for {topics[i%len(topics)]} (variant {i}). Include type hints or types, docstrings/comments, error handling and unit tests."
    body={"model":"dsv41-flash-1017","messages":[{"role":"user","content":p}],"max_tokens":mt,"stream":False}
    req=urllib.request.Request(BASE,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    t=time.time(); r=json.load(urllib.request.urlopen(req,timeout=600)); e=time.time()-t
    res[i]=(r["usage"]["completion_tokens"],e)
for n in (4,16):
    res=[None]*n; th=[threading.Thread(target=run,args=(i,res,1024)) for i in range(n)]
    t=time.time(); [x.start() for x in th]; [x.join() for x in th]; w=time.time()-t
    toks=sum(r[0] for r in res); per=[r[0]/r[1] for r in res]
    print(json.dumps({"concurrency":n,"wall_s":round(w,2),"total_tokens":toks,"aggregate_tok_s":round(toks/w,1),"median_req_tok_s":round(st.median(per),1),"min_req_tok_s":round(min(per),1)}))
