# Evidence

These JSON and text files are captured server information and experiment
reports. They are retained as data, not as executable code. File names identify
the serving profile and test used to produce each report.

Reports can contain host-specific paths and configuration details. Do not treat
them as proof that the packaged candidate is currently installed; verify the
live service and `/server_info` before making changes.

`dp4-runtime-comparison-20261013T175226Z.json` links the unchanged-profile
short, four-user 100k/260k, same-worker host-restore and post-load tool tests.
Short prompts/outputs match the historical speed-only test, but profile
changes are not causally isolated. Long six-user historical references have
different concurrency and are explicitly descriptive, not matched A/Bs.

`dp4-400k-validation-summary-20261013T204840Z.json` links installed-400k startup,
scoped single/four independent 398k tests and rank0 host restoration.
`opencode-gateway-dp-header-probe-20261013T204440Z.json` is a separate failed
public routing qualification: its neutral request did not register on this
engine. Neither that probe nor the successful private tests validates the
workstation/gateway path or a deployed affinity plugin/queue reduction.
