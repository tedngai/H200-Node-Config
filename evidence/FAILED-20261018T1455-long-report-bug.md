# Failed/invalid probe — 2026-10-18 14:55 UTC

## Long-context report bug

The first calibrated long-context attempt completed the HTTP request but the
test script crashed while writing the report:

```text
TypeError: 'in <string>' requires string as left operand, not int
  File "tests/openai_probe.py", line 303, in <genexpr>
    "recalled": all(c in content for c in codes.values()),
```

Cause: `codes` includes the integer `approx_chars`; the recall check iterated
all values instead of the three marker codes.

The report was not written, so the request result is not counted as evidence.
The script was fixed to iterate only the three string codes. The next probe was
also used to expose a length-calibration issue: the first successful report
(`long-100k-20261018T145525Z.json`) had only **74,921** prompt tokens, not
100k. That report is retained but is not the 100k gate.

The calibrated rerun (`long-100k-20261018T145556Z.json`) used the server's
`/tokenize` endpoint and reached **100,001** prompt tokens with correct recall.
