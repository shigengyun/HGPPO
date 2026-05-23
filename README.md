# HGPPO: Hierarchical GA-PPO Scheduling for Dynamic Hybrid Disassembly Lines

![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue) ![stable-baselines3](https://img.shields.io/badge/stable--baselines3-2.0%2B-green) ![SimPy](https://img.shields.io/badge/SimPy-4.0-orange) ![Gymnasium](https://img.shields.io/badge/Gymnasium-0.26%2B-yellow)

## Overview

HGPPO is a hybrid scheduling framework combining Proximal Policy Optimization (PPO) for global workstation selection with an Adaptive Genetic Algorithm (GA) for local task prescreening. The environment simulates a flexible hybrid disassembly line with 12 workstations (automated + manual), stochastic failures, and dynamic product arrivals. Published in Journal of Manufacturing Systems.

## Repository Structure

```
git-gjs/
├── run.py                  # Main training script (HGPPO)
├── evaluate.py             # Load and evaluate a trained model
├── plot_results.py         # Plot training reward curves
├── run_scpr_ablation.py    # PPO+SCPR ablation baseline
├── logger.py               # Logging utilities
├── config/
│   ├── ppo1.json           # PPO hyperparameters
│   ├── ga_config.json      # GA hyperparameters
│   ├── a2c1.json           # A2C config (for baselines)
│   ├── dqn1.json           # DQN config (for baselines)
│   └── environment.json    # Environment settings
└── production/
    ├── __init__.py
    └── envs/
        ├── production_env.py    # Gymnasium environment wrapper
        ├── transport.py         # HGPPO agent (PPO+GA decision)
        ├── initialize_env.py    # Environment parameter setup
        ├── reward_functions.py  # Reward computation
        ├── machine.py           # Workstation simulation (SimPy)
        ├── source.py            # Product source (arrivals)
        ├── sink.py              # Product sink (completion)
        ├── order.py             # Disassembly order/task
        ├── heuristics.py        # Baseline dispatching rules
        ├── time_calc.py         # Time & normalization utilities
        ├── Job_ProcessTime.py   # Processing time distributions
        └── logging_config.py   # Logging setup
```

## Requirements

```
python>=3.9
simpy==4.0
gymnasium>=0.26
stable-baselines3>=2.0
torch>=2.0
numpy>=1.24
matplotlib>=3.7
tqdm>=4.66
scipy>=1.11
```

## Installation

```bash
git clone https://github.com/shigengyun/HGPPO.git
cd HGPPO
pip install -r requirements.txt
```

## Quick Start

```bash
# Train HGPPO (saves model to model_history2/)
python run.py

# Evaluate a trained model (edit model path in evaluate.py)
python evaluate.py

# Run PPO+SCPR ablation baseline
python run_scpr_ablation.py

# Plot reward curves (after training)
python plot_results.py
```

## Configuration

### `config/ppo1.json` — PPO hyperparameters

| Field | Value | Description |
|-------|-------|-------------|
| `policy` | `"MlpPolicy"` | Network architecture |
| `learning_rate` | `5e-4` | Adam optimizer learning rate |
| `n_steps` | `256` | Rollout buffer size |
| `batch_size` | `256` | Mini-batch size |
| `n_epochs` | `10` | PPO update epochs per rollout |
| `gamma` | `0.87` | Reward discount factor |
| `gae_lambda` | `0.87` | GAE lambda |
| `clip_range` | `0.2` | PPO clipping range |
| `target_kl` | `0.01` | Early stopping KL threshold |

### `config/ga_config.json` — GA hyperparameters

| Field | Value | Description |
|-------|-------|-------------|
| `USE_GA` | `true` | Enable GA prescreening |
| `GA_LIGHTWEIGHT` | `true` | Use lightweight (fast) GA mode |
| `population_size` (lightweight) | `4` | GA population in lightweight mode |
| `generations` (lightweight) | `2` | Evolutionary iterations in lightweight mode |
| `population_size` (full) | `10` | GA population in full mode |
| `generations` (full) | `5` | Evolutionary iterations in full mode |
| `ga_selection_ratio` | `0.6` | Target fraction of orders to retain |
| `USE_SCPR_ONLY` | `false` | Set `true` for PPO+SCPR ablation |

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Episodes | 2500 | Training episodes |
| Timesteps/episode | 320 | Steps per episode |
| PPO lr | 5e-4 | Adam learning rate |
| γ (discount) | 0.87 | Reward discount |
| GA pop size | 4–10 | Lightweight/full mode |
| GA generations | 2–5 | Evolutionary iterations |
| Workstations | 12 | Physical stations (+ 1 virtual warehouse) |
| State dim | 29 | S₁(13) + S₂(13) + S₃(3) |

## Results

```
Method          | Mean Reward    | Flow Time
----------------|----------------|----------
HGPPO (ours)    | 153.31 ± 0.66  | ~97 s
PPO+FIFO        | 146.61 ± 0.80  | ~105 s
PPO+SCPR        | 145.89 ± 0.36  | ~106 s
PPO+RANDOM      | 141.91 ± 1.12  | ~112 s
DQN+FIFO        | 133.90 ± 3.50  | ~116 s
```

## License

MIT
