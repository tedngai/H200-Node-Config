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
