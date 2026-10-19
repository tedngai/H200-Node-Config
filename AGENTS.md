# AGENTS.md — DeepSeek-V4.1-Flash H200 test project

This file adds project-specific instructions to the global
`/home/advtech/AGENTS.md`. Read the global file first.

## Scope

This repository documents and tests `deepseek-ai/DeepSeek-V4.1-Flash-1017` on
the 8xH200 node using the cookbook H200 Flash-1017 FP4 low-latency recipe with
DSPARK, engram host tables and HiCache. It currently serves the client-facing
**8012 / pratt-high** endpoint with the scoped-tested phase 2b profile. A
production boot candidate is prepared, but the results here are not a
production quality/SLA qualification.

The validated GLM-5.3 service and its rollback path must be preserved. Do not
change `/etc/systemd/system/sglang-glm53.service` or the public alias as a side
effect of work in this repository.

Current measured state (phase 2b, 2026-10-19): installed on **8012 /
pratt-high** with `context_length=409600`, `--prefill-decode-interval 16`,
`--chunked-prefill-size 4096` and default tails. Mixed 4-warm/4-cold decode is
**292.4 tok/s** (ratio **0.649**, 9.33× phase 2a's 31.3). The cold-TTFT gate
passed (74.7 s single; four-cold max 268.3 s / wall 268.5 s). Costs: c16 decode
is 643.4 tok/s (0.974× phase 2a), 16 × 400k cold wall is 1068.9 s, and the
2048 replay wall is 81.6 s due to a straggler. Phase 2c tails is a separate
candidate. Do not claim production readiness or sustained load. See
`docs/RESULTS.md`, `docs/DECODE_PROFILE.md` and
`docs/SCHEDULER_OVERLAP_TEST.md`.

**When resuming after a restart, read `HANDOFF.md` first.** It records the
live unit state, phase 2b evidence, the separate tails runbook, rollback, and
the exact constraints for the next session.

## Git repository

- Remote: `https://github.com/tedngai/H200-Node-Config.git`
- Working branch: **`dsv41-flash-1017`**
- Do **not** push this project's history to `main`; `main` belongs to the
  existing H200 node configuration repository and must remain untouched.
- Commit all project changes before pushing. Keep the working tree clean.
- The installed unit is `/etc/systemd/system/sglang-dsv41.service`; the
  committed source of truth is
  `configs/systemd/candidates/sglang-dsv41-8012-pratt-high-mr16-nodelay-ctx409600-overlap2b.service`.
  They currently match. The 8012 phase 2a unit is the exact fallback that
  preserves the endpoint; the 8013 phase 1/mr16 units are historical test
  endpoints. The production boot candidate is
  `configs/systemd/candidates/sglang-dsv41-8012-pratt-high-mr16-nodelay-ctx409600-overlap2b-production.service`,
  prepared but not installed. Record any intentional divergence.

## GitHub credentials

A GitHub personal access token for account **`tedngai`** is stored outside this
repository at:

```text
/home/advtech/glm53-inference/.env
```

The variable name is:

```text
GITHUB_PAT
```

The `.env` file is outside this repository, is ignored and untracked in
`glm53-inference`, and must stay that way.

**Never** print, log, copy, commit, or place the token value on a command line
or in a tracked file. Do not add it to this repository, `.git/config`, a remote
URL, shell history, or an evidence file. Read it only inside a process that
needs it for authentication.

## Push procedure

Run from this repository. This uses a temporary `GIT_ASKPASS` helper, keeps the
token out of argv and git config, and removes the helper on exit.

```bash
cd /home/advtech/dsv41-flash-inference

set -eu
set +x

ENV_FILE=/home/advtech/glm53-inference/.env
[ -f "$ENV_FILE" ] || { echo "missing $ENV_FILE"; exit 1; }

set -a
. "$ENV_FILE"
set +a
[ -n "${GITHUB_PAT:-}" ] || { echo "GITHUB_PAT is empty"; exit 1; }

ASKPASS=$(mktemp /tmp/opencode/git-askpass.XXXXXX)
chmod 700 "$ASKPASS"
cat >"$ASKPASS" <<'EOF'
#!/bin/sh
case "$1" in
  Username*) printf '%s\n' "tedngai" ;;
  Password*) printf '%s\n' "$GITHUB_PAT" ;;
  *) exit 1 ;;
esac
EOF
trap 'rm -f "$ASKPASS"' EXIT

GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 \
  git push origin dsv41-flash-1017
```

After pushing, verify the branch and clean state:

```bash
git status --short --branch
git ls-remote --heads origin dsv41-flash-1017
```

If authentication fails, report the error but do not print the token. Do not
retry by putting the token in the URL or in git config.

## Validation and operational notes

- Use `python3 -m py_compile` on changed test scripts before committing.
- Keep `docs/` and `README.md` consistent with measured results.
- The service is not enabled and uses `Restart=no`; do not enable or restart it
  without explicit approval. The prepared production candidate adds bounded
  crash restart and boot enablement, but the user runs privileged
  install/enable/restart commands.
- Record effective flags, startup time, memory headroom, errors and the
  distinction between cold prefill, GPU-cached reuse, host-cache restoration
  and decode.
- Do not run privileged install/restart commands; the user runs those.
- Do not claim sustained production load or broad coding quality from the
  scoped tests in this repository.
