# Systemd configurations

`sglang-glm53.service` is a copy of `/etc/systemd/system` at the latest recorded
deployment. Always verify the actual unit and API before treating it as live.

Installed and ready: `candidates/dp4-mtp-hicache128-400k.service`
raised only context to 400,000 including output; four active requests, DP4,
MTP, HiCache and queue settings remain fixed. Exact narrow rollback:
`candidates/pre-400k-dp4-mtp-hicache128.service` (scoped-tested prior 262k DP4).
The root snapshot now matches the authoritative live 400k unit. Scoped single
and four independent 398k tests passed. Queues remain eight per worker; the
approved two-per-worker trial follows client affinity/retry verification and
is not installed. See `docs/NEXT_EXPERIMENT.md` for status and rollback commands.

The files in `candidates/` are historical or prepared alternatives. They are
not interchangeable without checking the corresponding evidence and context
documents. `optional/sglang-glm53-flash.service` serves the separate
GLM-5.3-Flash checkpoint and is not the full GLM-5.3 service.

All units are host-specific: review Docker image digests, cache mounts,
service conflicts, port, model alias, GPU topology, and memory settings before
installation.
