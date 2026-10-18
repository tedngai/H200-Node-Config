# AGENTS.md — DeepSeek-V4.1-Flash H200 test project

This file adds project-specific instructions to the global
`/home/advtech/AGENTS.md`. Read the global file first.

## Scope

This repository documents and tests `deepseek-ai/DeepSeek-V4.1-Flash-1017` on
the 8xH200 node using the cookbook H200 Flash-1017 FP4 low-latency recipe with
DSPARK, engram host tables and HiCache. It is an isolated test on port **8013**
with served name **`dsv41-flash-1017`**; it is not the production `pratt-high`
service on port 8012.

The validated GLM-5.3 service and its rollback path must be preserved. Do not
change `/etc/systemd/system/sglang-glm53.service` or the public alias as a side
effect of work in this repository.

Current measured state: 16 concurrent 400k requests complete, but peak observed
was **14 running + 1–2 queued**, not 16 simultaneous active. Do not claim
16-active or production readiness without a new test. See `docs/RESULTS.md`.

## Git repository

- Remote: `https://github.com/tedngai/H200-Node-Config.git`
- Working branch: **`dsv41-flash-1017`**
- Do **not** push this project's history to `main`; `main` belongs to the
  existing H200 node configuration repository and must remain untouched.
- Commit all project changes before pushing. Keep the working tree clean.
- The installed unit is `/etc/systemd/system/sglang-dsv41.service`; the
  committed source of truth is
  `configs/systemd/candidates/sglang-dsv41-8013-mr16.service`. They currently
  match. Record any intentional divergence.

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
- The service is not enabled and uses `Restart=no`; do not enable it without
  approval.
- Record effective flags, startup time, memory headroom, errors and the
  distinction between cold prefill, GPU-cached reuse, host-cache restoration
  and decode.
- Do not run privileged install/restart commands; the user runs those.
- Do not claim sustained production load or broad coding quality from the
  scoped tests in this repository.
