# MAMBA with SMAX eval support

A scalable multi-agent model-based reinforcement learning framework supporting multiple environments including **SMAX** (newly integrated) and StarCraft.

> **Note**: This fork has integrated SMAX environment and **completely removed Flatland support** to streamline the codebase for latest python and python package support.

## 🆕 What's New

- **SMAX Integration**: Added support for SMAX (StarCraft Multi-Agent Challenge) environment via JaxMARL
- **Flatland Removed**: Completely removed Flatland environment, configs, and bundled flatland-2.2.2 directory to reduce repository size and focus on StarCraft scenarios
- **Enhanced Logging**: Added comprehensive print statements, wandb logging, and PKL data export for research analysis
- **Model Persistence**: Automatic model checkpointing and training data storage for reproducibility

## Installation

**Requirements**: Python 3.10+

### Installation: Conda Environment

Create and activate the conda environment with all dependencies:

```bash
conda env create -f environment.yml
conda activate mamba
```

### Environment-Specific Setup

**For SMAX** (NEW):
```bash
pip install "jax[cpu]==0.4.31"   
pip install jaxmarl
```

**For StarCraft (SMAC)**:
Follow the installation guide at [https://github.com/oxwhirl/smac#installing-starcraft-ii](https://github.com/oxwhirl/smac#installing-starcraft-ii)


## Usage

### Quick Start

**SMAX Training** (Recommended):
```bash
python train.py --env smax --env_name 3m --n_workers 4 --seed 1 --steps 1000000 --mode online
```

**StarCraft Training**:
```bash
python train.py --env starcraft --env_name 3m --n_workers 2 --seed 1 --steps 1000000 --mode online
```


### Command Line Arguments

| Argument | Description | Default | Example |
|----------|-------------|---------|---------|
| `--env` | Environment type | `smax` | `smax`, `starcraft` |
| `--env_name` | Specific scenario/map | `5_agents` | `3m`, `5m_vs_6m`, `8m` |
| `--n_workers` | Number of parallel workers | `2` | `4`, `8` |
| `--seed` | Random seed | `1` | `42`, `123` |
| `--steps` | Total training steps | `1000000` | `2000000` |
| `--mode` | Wandb logging mode | `disabled` | `online`, `offline`, `disabled` |

### SMAX Scenarios

Available SMAX maps (via JaxMARL):
- `3m` - 3 Marines vs 3 Marines
- `5m_vs_6m` - 5 Marines vs 6 Marines  
- `8m` - 8 Marines vs 8 Marines
- And more scenarios from JaxMARL SMAX

For detailed SMAX usage, see [SMAX_QUICKSTART.md](SMAX_QUICKSTART.md)

## Supported Environments

### 1. SMAX (NEW) 🆕

JAX-based StarCraft Multi-Agent Challenge for fast, scalable training.

**Features**:
- CPU-optimized JAX execution
- Full SMAC scenario compatibility
- Efficient parallel training
- Discrete action spaces

### 2. SMAC (StarCraft II)

<img height="300" alt="starcraft" src="https://user-images.githubusercontent.com/22059171/152656435-1634c15b-ca6d-4b23-9383-72fe3759b9e3.png">

Multi-agent cooperative scenarios in StarCraft II.

**Environment**: [https://github.com/oxwhirl/smac](https://github.com/oxwhirl/smac)

## Output & Logging

### Training Outputs

All training artifacts are saved to `{date}_results/{env}/{scenario}/run{N}/`:

```
1211_results/smax/3m/run1/
├── ckpt/
│   ├── model_200Ksteps.pth
│   ├── model_400Ksteps.pth
│   └── model_final.pth
├── agent/          # Backed up code
├── configs/        # Backed up configs
└── networks/       # Backed up networks
```

### Research Data (PKL Files) 📊

Training metrics are automatically saved to PKL files for plotting:

**Location**: `{date}_results/{env}/mamba_{scenario}_seed{seed}.pkl`

**Contents**:
```python
{
    'steps': [1000, 2000, ...],           # Evaluation steps
    'eval_win_rates': [0.5, 0.6, ...],    # Win rates / scores
    'eval_returns': [100, 120, ...]       # Cumulative returns
}
```

**Usage Example**:
```python
import pickle
import matplotlib.pyplot as plt

with open('1211_results/smax/mamba_3m_seed123.pkl', 'rb') as f:
    data = pickle.load(f)

plt.plot(data['steps'], data['eval_win_rates'])
plt.xlabel('Training Steps')
plt.ylabel('Win Rate')
plt.savefig('learning_curve.png')
```

### Wandb Integration

Logged metrics include:
- `win` / `reward` / `scores` (environment-dependent)
- `returns` - Episode cumulative rewards
- `eval_win_rate` - Evaluation win rate (every 1000 steps)
- `eval_returns` - Evaluation returns
- `Agent/Returns`, `Agent/val_loss`, `Agent/actor_loss`

## Code Structure

```
mamba/
├── agent/               # MAMBA implementation
│   ├── controllers/     # Inference logic
│   ├── learners/        # Training logic (with logging & PKL export)
│   ├── memory/          # Replay buffer
│   ├── models/          # MAMBA architecture
│   ├── optim/           # Loss optimization
│   ├── runners/         # Multi-worker coordination (with eval & save)
│   ├── utils/           # Helper functions
│   └── workers/         # Environment interaction
├── configs/             # Configuration files
│   ├── dreamer/         # Dreamer configs
│   │   ├── smax/        # SMAX-specific configs (NEW)
│   │   └── optimal/     # Optimal hyperparameters
│   └── flatland/        # Flatland configs
├── env/                 # Environment wrappers
│   ├── smax/            # SMAX environment (NEW)
│   ├── starcraft/       # SMAC environment
│   └── flatland/        # Flatland environment
└── networks/            # Neural network architectures
```

## Configuration

### Optimal Parameters

To use optimal parameters from the paper:
1. Navigate to `configs/dreamer/optimal/`
2. Copy desired config to:
   - [DreamerAgentConfig.py](configs/dreamer/DreamerAgentConfig.py)
   - [DreamerLearnerConfig.py](configs/dreamer/DreamerLearnerConfig.py)

### SMAX Configuration

Edit `configs/dreamer/smax/SMAXLearnerConfig.py` to adjust:
- Learning rates (`MODEL_LR`, `ACTOR_LR`, `VALUE_LR`)
- Buffer capacity (`CAPACITY`)
- Training intervals (`N_SAMPLES`)
- Epochs (`MODEL_EPOCHS`, `PPO_EPOCHS`)

## Advanced Features

### Multi-Seed Experiments

```bash
for seed in 1 2 3 4 5; do
    python train.py --env smax --env_name 3m --n_workers 4 --seed $seed --steps 1000000 --mode online
done
```

### Custom Training Intervals

Modify save intervals in training command (default: 200K steps):
```python
# In train.py, train_dreamer function
runner.run(exp.steps, exp.episodes, save_interval=100000, save_mode='interval')
```

### Validation Script

Test the SMAX integration:
```bash
./test_smax_integration.sh
```

Expected output: `✓ All checks passed!`

## Documentation

- **[SMAX_INTEGRATION.md](SMAX_INTEGRATION.md)** - Detailed SMAX integration documentation
- **[SMAX_QUICKSTART.md](SMAX_QUICKSTART.md)** - Quick start guide for SMAX
- **[test_smax_integration.sh](test_smax_integration.sh)** - Integration validation script

## Key Changes

✅ **Added**: SMAX environment support via JaxMARL  
✅ **Added**: Comprehensive logging (print, wandb, PKL export)  
✅ **Added**: Automatic model checkpointing every 200K steps  
✅ **Added**: Evaluation every 1000 steps with 10 episodes  
✅ **Removed**: Complete Flatland support (environment code, configs, and bundled flatland-2.2.2 directory)  
✅ **Enhanced**: Multi-environment support (SMAX, SMAC)  
✅ **Enhanced**: Training data persistence for research analysis  
✅ **Focused**: Streamlined codebase for StarCraft-based multi-agent research  

## Citation

If you find the **SMAX integration** in this fork useful for your research, please consider citing:

```bibtex
@misc{zhang2024mamba_smax,
    author = {Zhang, Yang},
    title = {MAMBA with SMAX Integration},
    year = {2024},
    publisher = {GitHub},
    howpublished = {\url{https://github.com/breez3young/mamba-smax}},
    note = {SMAX environment integration with enhanced logging and data export for multi-agent reinforcement learning research}
}
```

**Acknowledgments**: This SMAX integration was independently developed to extend MAMBA's capabilities to the JaxMARL SMAX environment, with added features for research reproducibility including comprehensive logging, automatic checkpointing, and training data export.

## Troubleshooting

**Issue**: SMAX environment fails to initialize  
**Solution**: Ensure JAX dependencies are installed: `pip install jax jaxlib jaxmarl`

**Issue**: CUDA errors with SMAX  
**Solution**: SMAX uses CPU-only JAX by default (configured in `env/smax/SMAX.py`)

**Issue**: Wandb login required  
**Solution**: Use `--mode disabled` to skip wandb, or run `wandb login`

For more help, see [SMAX_QUICKSTART.md](SMAX_QUICKSTART.md) or open an issue.
