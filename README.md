# GLM-5.3 inference project

Repository-ready package of the GLM-5.3 SGLang deployment configurations,
validation scripts, experiment evidence, and operational notes from the H200
node.

## Contents

- `configs/systemd/sglang-glm53.service` — copy of the authoritative configured
  full `zai-org/GLM-5.3` service unit at packaging time.
- `configs/systemd/candidates/` — FP8/MTP, no-MTP, HiCache, and installed DP4
  service candidates.
- `configs/systemd/optional/` — separate GLM-5.3-Flash service unit.
- `tests/` — benchmark, near-limit retrieval/tool-call smoke, concurrency, and
  HiCache restoration scripts.
- `evidence/` — captured server configurations and benchmark/validation JSON.
- `docs/` — experiment history, results, context, and the DP4 handoff.

The model weights, Hugging Face cache, Docker images, live logs, and host shell
history are intentionally not included. They are large or host-specific and
should not be committed to Git.

## Captured deployment state

The installed and ready unit is TP8 + DP4 attention with lightweight EAGLE
MTP, HiCache 128 GB per rank, a 262,144-token context, static memory fraction
`0.90`, four running requests, and eight queued requests (launch setting).
Startup verification found **472,768 GPU token slots per DP worker**, about
**1,891,072 unique slots across four private pools**, and one active request
allowed per worker. Four concurrent 260k retrievals, post-load tools/reasoning
and same-worker DP0 HiCache restoration passed; no queueing/retractions. See
`docs/NEXT_EXPERIMENT.md` and `docs/results.md` for the qualification status.

The prior validated production profile (TP8/DP1, MTP off, HiCache 128, six
running requests) is retained in
`configs/systemd/candidates/fp8-no-mtp-6-hicache128.service` and its evidence.

The API was not ready during initial packaging but scoped runtime tests now
passed. Four warmed 260k sessions ran at **352 aggregate / 89.6 per-request
tok/s**. HiCache restored **259,968 host tokens with zero device hits** in
**3.29 s versus 53.80 s cold**. Short c1/c4 rates (93.5 / 336.6 aggregate)
were below historical TP-only MTP, so DP4 is not an automatic speed upgrade;
this does not isolate HiCache overhead. Sustained coding load, six/eight users,
host restoration on DP1-3 and 400k remain unqualified.
See `evidence/dp4-runtime-comparison-20261013T175226Z.json` and `docs/results.md`.

## Install a service unit

These commands require root access and restart the full model on all GPUs. Drain
requests first and verify the selected unit before restarting:

```bash
sudo install -o root -g root -m 0644 \
  configs/systemd/sglang-glm53.service \
  /etc/systemd/system/sglang-glm53.service
sudo systemctl daemon-reload
sudo systemctl restart sglang-glm53.service
```

The service files contain this host's Docker image, model-cache mount, model
alias, port, and GPU topology. Adapt those values before using them elsewhere.

## Run checks

The scripts use only Python's standard library on the host and expect a live
SGLang server at `http://127.0.0.1:8012`, with the tokenizer available inside
the `sglang-glm53` container:

```bash
python3 -m py_compile tests/*.py
python3 -B -m unittest discover -s tests -p 'test_dp4_profile.py' -v
python3 -B tests/dp4_smoke_test.py
python3 -B tests/dp4_profile_test.py --mode short --concurrency 4 --output-tokens 256
```

Long-context and HiCache tests are intentionally resource-intensive. Use the
commands and admission guards documented in `docs/test-history.md` and
`docs/NEXT_EXPERIMENT.md` rather than starting them casually.
The older concurrency/HiCache helpers are TP-only; do not use them unchanged
to qualify the live DP4 profile. `dp4_profile_test.py` handles private-pool
guards, rank pinning, per-worker metrics and same-worker cache eviction.
`docs/NEXT_EXPERIMENT.md` has the long/host commands. Run one test at a time
on an idle server; no helper changes the service or flushes caches.

## Git setup

This directory is an initialized Git repository. After any change, commit it:

```bash
cd glm53-inference
git status
git add <changed-files>
git commit -m "Describe the change"
```

See `AGENTS.md` for the repository workflow rules, including the requirement
to always commit changes and keep `docs/` copies and the package manifest in
sync.
