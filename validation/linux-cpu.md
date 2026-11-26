# Release validation

Linux x86-64 CPU, CPython 3.13.5, PyTorch 2.10.0+cpu, NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0, pytest 9.0.2.

| Command | Result | Log |
|---|---|---|
| `python verify_integrity.py` | PASS | Repository SHA-256 manifest |
| `python run.py check` | PASS | [Native check](linux-cpu-check.log) |
| `python run.py demo` | PASS | [Native demo](linux-cpu-demo.log) |

Selected checks exercise the documented native entry points.

The archival preparation logs remain alongside these results. The release audit ran the documented inference or benchmark paths and did not retrain all model weights. GPU, Windows and macOS were outside the verification scope.
