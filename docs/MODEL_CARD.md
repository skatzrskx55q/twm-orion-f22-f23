# Model card — ORION F22–F23 — Adaptive Causal Dynamics

## Intended use
Controlled synthetic world-dynamics research, reproduction, ablation and extensions. Not validated for real-world forecasts, production automation, robotics, financial or safety decisions.

## Design
F22 learns continuous six-node graph dynamics; F23 AdaptiveOddLink estimates nonlinear response from 96 observed transitions. Unlike NC3 it predicts numerical updates rather than selecting masks.

## Evaluation
New response forms H90: F23 90/96 vs F22 33/96 and F21-P2 0/96. New topology + response H90: 69/72 vs 32/72 and 0/72. F22 native family 63/64 vs 0/64.

## Known limitations
Fixed six nodes; no creation/deletion of edges or arbitrary new operations. Earlier controls received the same 96 observations but were not trained with that support budget.

## Data and weights
F22 and F23 checkpoints and the frozen upstream F21 package are included.

The original protocols define split, features and seeds. See the [release validation](../validation/linux-cpu.md).

## Technical entry points
[code/orion_f22.py](../research/TWM_ORION_F22_F23/code/orion_f22.py) | [code/orion_f23.py](../research/TWM_ORION_F22_F23/code/orion_f23.py) | [code/reproduce.py](../research/TWM_ORION_F22_F23/code/reproduce.py) | [checkpoints/](../research/TWM_ORION_F22_F23/checkpoints/) | [docs/TECHNICAL_REPORT_RU.md](../research/TWM_ORION_F22_F23/docs/TECHNICAL_REPORT_RU.md) | [results/](../research/TWM_ORION_F22_F23/results/) | [upstream_F21.zip](../research/TWM_ORION_F22_F23/upstream_F21.zip)

## Evidence status
Historical status: Accepted task-specific continuous reference baseline. Fresh selected check: PASS. See `../validation/current_check.log`. Only the documented research environment is evaluated; there is no warranty of generalization.
