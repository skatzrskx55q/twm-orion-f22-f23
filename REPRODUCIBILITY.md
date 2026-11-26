# Reproducibility

1. Install the pinned Python dependencies using [QUICKSTART.md](QUICKSTART.md).
2. Verify the source SHA-256 manifest with `python verify_integrity.py`.
3. Run the frozen native check. The historical command is `python code/reproduce.py --scope smoke`.
4. Run `python run.py --keep ../twm-orion-f22-f23-repro demo` to retain outputs and compare them against [archived results](research/TWM_ORION_F22_F23/results/SUMMARY.json). The direct native demo command is `python code/reproduce.py --scope all`.

## Training or calibration

The F23 standalone retraining script fixes a new explicit random seed. Use the original protocol for comparison with archived runs.

```bash
python run.py --keep ../twm-orion-f22-f23-training native -- code/retrain.py --seed 1701 --steps 1800
```

[original guide](research/TWM_ORION_F22_F23/README_START_HERE_RU.md) · [Original release guide](docs/REPRODUCTION.md) · [Release validation](validation/linux-cpu.md)
