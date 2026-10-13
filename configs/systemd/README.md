# Systemd configurations

`sglang-glm53.service` is copied from `/etc/systemd/system` at packaging time
and represents the authoritative live full-model unit.

The files in `candidates/` are historical or prepared alternatives. They are
not interchangeable without checking the corresponding evidence and context
documents. `optional/sglang-glm53-flash.service` serves the separate
GLM-5.3-Flash checkpoint and is not the full GLM-5.3 service.

All units are host-specific: review Docker image digests, cache mounts,
service conflicts, port, model alias, GPU topology, and memory settings before
installation.
