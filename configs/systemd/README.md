# Systemd configurations

`sglang-glm53.service` is copied from `/etc/systemd/system` at packaging time
and represents the authoritative live full-model unit.

Approved next, pending user install: `candidates/dp4-mtp-hicache128-400k.service`
raises only context to 400,000 including output; four active requests, DP4,
MTP, HiCache and queue settings remain fixed. Exact narrow rollback:
`candidates/pre-400k-dp4-mtp-hicache128.service` (scoped-tested live 262k DP4).
The root snapshot is not replaced until the authoritative live unit and API
verify a deployment change. See `docs/NEXT_EXPERIMENT.md` for commands.

The files in `candidates/` are historical or prepared alternatives. They are
not interchangeable without checking the corresponding evidence and context
documents. `optional/sglang-glm53-flash.service` serves the separate
GLM-5.3-Flash checkpoint and is not the full GLM-5.3 service.

All units are host-specific: review Docker image digests, cache mounts,
service conflicts, port, model alias, GPU topology, and memory settings before
installation.
