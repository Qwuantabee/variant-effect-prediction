# Setup notes

Detail behind the short Setup section of the [README](../README.md).


## Full setup notes

```
pip install -e ".[dev]"
pytest -m "not slow"        # 273 tests, no model needed
pytest -m slow              # downloads an 8M checkpoint AND ~1.6 GB of ProteinGym archives
```

**The assay data is not committed** (`data/` is git-ignored). The slow tests fetch
three ProteinGym archives into it on first use, measured on a fresh clone:
substitutions 43 MB, MSAs 1.50 GB, AlphaFold2 structures 19 MB. On a fresh clone
the nine tests that read real assays fail
with "assays not downloaded" rather than skipping, because a missing input you
did not ask to miss is a failure, not a pass. Either fetch the data or run the
suite on a machine without it using the explicit escape. The fetch has to name
all 217 assays (about 1 GB extracted); a bare `data.ensure_assays()` extracts
only the eight default assays and leaves the same nine tests failing:

```
python -c "from variant_effect import data; data.ensure_assays(list(data.reference_table()['DMS_id']))"
```

```
pytest                      # 288 passed, with the data
```

The escape:

```
VE_ALLOW_MISSING_DATA=1 pytest -m "not slow"    # 265 passed, 8 skipped, no data/
VE_ALLOW_MISSING_DATA=1 pytest                  # 279 passed, 9 skipped, no data/
```

Every test that needs an assay, an MSA or a results file now guards on that file
and skips under the escape instead of raising. That was not true before
2026-11-15: six tests called the loader without a guard, so the documented escape
still produced failures. If you add a test that reads data, use
`conftest.require(...)` - a bare `open()` turns a missing download into a crash.

Without the escape, on an empty `data/`: 9 failed, 279 passed. That is the
intended behaviour - it tells you the data is missing rather than quietly
reporting a smaller green run.

The suite is fully green on the machine that produced these results (288 passed)
and that is the state the numbers below refer to.

Two environment notes, both of which cost time to work out:

- `HF_HUB_OFFLINE=1` after the first download. Without it, `from_pretrained`
  makes a hub request that can hang for tens of minutes on some networks even when
  the model is fully cached; loading the 650M model went from hanging indefinitely
  to 10 seconds.
- `pip_system_certs` breaks `huggingface_hub` with a `RecursionError` deep in
  `ssl.py`. If downloads fail that way, uninstall it from the environment.
