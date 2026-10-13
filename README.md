# GLM-5.3 inference project

Repository-ready package of the GLM-5.3 SGLang deployment configurations,
validation scripts, experiment evidence, and operational notes from the H200
node.

## Contents

- `configs/systemd/sglang-glm53.service` — copy of the authoritative configured
  full `zai-org/GLM-5.3` service unit at packaging time.
- `configs/systemd/candidates/` — FP8/MTP, no-MTP, HiCache, and pending DP4
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
allowed per worker. Four small rank-pinned probes and OpenAI tools/separated
reasoning passed; metrics confirmed four active with no queueing/retractions. See
`docs/NEXT_EXPERIMENT.md` and `docs/results.md` for the qualification status.

The prior validated production profile (TP8/DP1, MTP off, HiCache 128, six
running requests) is retained in
`configs/systemd/candidates/fp8-no-mtp-6-hicache128.service` and its evidence.

The API was not ready during initial packaging but readiness and small smoke
are now verified. Near-limit/sustained-load and DP4+MTP host-restoration gates
remain pending. No 400k context is enabled or qualified.
See `evidence/dp4-startup-validation-summary.json` and `docs/NEXT_EXPERIMENT.md`.

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
python3 tests/benchmark.py --label local-benchmark
python3 tests/dp4_smoke_test.py
python3 tests/smoke_test.py --label local-chat-smoke --context-tokens 260000
```

Long-context and HiCache tests are intentionally resource-intensive. Use the
commands and admission guards documented in `docs/test-history.md` and
`docs/NEXT_EXPERIMENT.md` rather than starting them casually.
The older concurrency/HiCache helpers are TP-only; do not use them unchanged
to qualify the live DP4 profile. The new DP4 helper tests only small requests.

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
