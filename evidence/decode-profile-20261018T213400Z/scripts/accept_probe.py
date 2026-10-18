import json, time, urllib.request, sys
BASE="http://127.0.0.1:8013/v1/chat/completions"
prompts=[
 "Write a Python module implementing an LRU cache with TTL expiry, thread safety via a lock, and a decorator interface. Include type hints, docstrings, and pytest unit tests covering eviction, expiry and concurrency.",
 "Refactor this design into idiomatic TypeScript: a REST client class with retry/backoff, request cancellation via AbortController, typed error classes, and a small usage example. Explain the key design choices briefly after the code.",
]
out=[]
for p in prompts:
    body={"model":"dsv41-flash-1017","messages":[{"role":"user","content":p}],"max_tokens":1536,"stream":False}
    req=urllib.request.Request(BASE,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    t=time.time(); r=json.load(urllib.request.urlopen(req,timeout=600)); e=time.time()-t
    u=r["usage"]; m=r["choices"][0]["message"]
    out.append({"elapsed_s":round(e,3),"prompt_tokens":u["prompt_tokens"],"completion_tokens":u["completion_tokens"],"tok_s":round(u["completion_tokens"]/e,2),"finish":r["choices"][0]["finish_reason"],"reasoning_chars":len(m.get("reasoning_content") or ""),"content_chars":len(m.get("content") or "")})
print(json.dumps(out,indent=1))
