# Reproduction guide

1. Follow installation in [README](../README.md). CPU only was tested in the release preparation environment.
2. `python verify_integrity.py` checks the byte-level hashes of distributed source artifacts without ML dependencies.
3. `python run.py check` invokes: `python code/reproduce.py --scope smoke` in `research/TWM_ORION_F22_F23`. Exit code is propagated, including failures.
4. `python run.py --keep ../experiment demo` invokes: `python code/reproduce.py --scope all`. The copied data and output remain in `../experiment`. Consult original README before long runs.
5. Training / extended experiment: `See code/retrain.py and docs/TECHNICAL_REPORT_RU.md / См. исходный отчёт`. Execute in a separate copied tree, preserve originals, and read the historical protocol and unlock rules. Do not tune using the released EXAM labels. Bash commands may require Linux/WSL.

## Metric definitions
New response forms H90: F23 90/96 vs F22 33/96 and F21-P2 0/96. New topology + response H90: 69/72 vs 32/72 and 0/72. F22 native family 63/64 vs 0/64. Interpret H90/H16 and exact final state versus full trajectory using the native protocol, not as a shared benchmark. NLL is lower-is-better. Read the original source reports for the denominators and negative controls.

## Fresh preparation check
`PASS`. Raw log is in `../validation/current_check.log`. The current check is not full end-to-end retraining, and `demo` is a documented native entry point, not necessarily newly executed in full during preparation.
