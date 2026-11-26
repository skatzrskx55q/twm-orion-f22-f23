# Contributing

For packaging, test and documentation patches, include the release tag, interpreter and dependency versions, the command used, and the resulting log or traceback. Run `python verify_integrity.py` and the selected check from [QUICKSTART.md](QUICKSTART.md). Record any known failure that you also reproduce.

For a research change, describe the question, controls, data splits and thresholds before running the experiment. Report seed-level metrics and negative outcomes. Include new experiment files in a separate change and preserve the provenance of the historical protocol, archived checkpoints and metrics in `research/`. Cite the original run and list third-party material and licensing changes with the contribution.
