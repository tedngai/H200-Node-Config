# Agent rules for the glm53-inference repository

Packaged GLM-5.3 SGLang deployment project: service units, validation
scripts, captured evidence, and documentation copies from the H200 node.

## Workflow

- **Always commit changes.** After any change to files in this repository,
  commit it with git using a descriptive message. Do not leave the working
  tree dirty at the end of a task.
- After adding, removing, renaming, or materially changing files, regenerate
  `docs/package-manifest.json` (paths, sizes, SHA-256) before committing.
- Keep the copies under `docs/` in sync with their authoritative sources in
  `/home/advtech`: `AGENTS.md`, `PROGRESS.xml`, `INFERENCE_CONTEXT.md`,
  `NEXT_EXPERIMENT.md`, `sglang-glm53-fp8-tests/results.md`, and
  `sglang-glm53-fp8-tests/README.md` (stored as `docs/test-history.md`).
- New experiment reports belong in `evidence/`, named by profile label and
  timestamp. Never overwrite an existing report with different content.
  Retain failed or invalid probes and mark them clearly.
- Never commit model weights, Hugging Face caches, Docker images, live logs,
  or shell history. `.gitignore` is a guardrail, not permission.

## Publishing

- The public GitHub remote is `https://github.com/tedngai/H200-Node-Config.git`
  (remote name `origin`). The local GitHub PAT is stored in the repository-root
  `.env` file under `GITHUB_PAT`; `.env` is ignored by Git.
- For pushes, read the PAT through a credential prompt or askpass helper. Never
  print it, put it in a remote URL or command history, copy it into tracked
  files, or stage `.env`. Verify `.env` remains untracked before committing.

## Live service

- The authoritative live unit is `/etc/systemd/system/sglang-glm53.service`,
  installed and restarted by the user with sudo. This repository records
  configurations and evidence; it does not deploy them.
- Verify the live service with `/server_info` before and after any change.
  Never claim a packaged candidate is deployed or qualified without runtime
  evidence recorded under `evidence/`.
- The older long-context and HiCache helpers are TP-only. DP4 validation requires
  rank-pinned requests (`routed_dp_rank`) and per-rank metric accounting;
  do not sum duplicate TP-rank gauges.
- `tests/dp4_profile_test.py` is the DP-aware short/long/host-restore helper;
  older `concurrency_test.py` and `hicache_test.py` remain TP-only. Run one
  experiment at a time on an idle engine. Keep speed-only forced-output
  benchmarks distinct from chat retrieval/tool checks and real coding quality.
