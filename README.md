# BBB-Penetrating Peptide Design with Enhanced RL Pipeline

A comprehensive machine learning pipeline for designing blood-brain barrier (BBB) penetrating peptides using reinforcement learning with enhanced strategies for small-sample optimization.

## Overview

This project implements a sophisticated RL-based peptide design pipeline with four key enhancements:

1. **Surrogate + RL Closed Loop**: RL optimizes against calibrated classifier probabilities
2. **KL Penalty**: Policy regularization against natural peptide distribution
3. **BO + RL Hybrid**: Bayesian optimization for active learning candidate selection
4. **High-Confidence Thresholding**: Success redefined using calibrated probability thresholds

## Quick Start

### Basic Pipeline
```bash
# Run full pipeline
python run_pipeline.py --target TFRC --config config.yaml

# Test individual components
python -m src.train_classifier --config config.yaml
python -m src.calibrate_temperature --config config.yaml
python -m src.train_rl_ppo --config config.yaml --target TFRC
python -m src.evaluate_candidates_gpu --config config.yaml --input runs/ppo/topk_candidates.csv
```

### Enhanced RL with Active Learning
```bash
# Run one active learning round
python active_loop_round.py --config config.yaml --target TFRC --round 1

# Use enhanced RL environment with KL penalty
python -m src.train_rl_ppo_enhanced --config config.yaml --target TFRC --use-kl

# Select candidates with Bayesian Optimization
python -m src.active_selection.bo_selector --config config.yaml --input candidates.csv --output selected.csv
```

## Enhanced Features

### 1. Calibrated Surrogate + RL Loop
- Uses temperature-scaled classifier probabilities for reward computation
- Integrates L1 baseline physicochemical scoring functions
- Confidence-based bonus system for high-probability sequences

### 2. KL Penalty for Policy Regularization
```yaml
rl_enhanced:
  kl_penalty:
    enabled: true
    beta: 0.01
    target_kl: 0.01
    adaptive: true
```

### 3. Bayesian Optimization Selection
- UCB acquisition function with uncertainty estimation
- MC Dropout for uncertainty quantification
- Diversity bonus to explore different sequence regions

### 4. High-Confidence Thresholding
```yaml
classifier:
  thresholds:
    high: 0.8
    medium: 0.5
    low: 0.3
```

## Project Structure

```
├── src/
│   ├── models/classifier.py          # LSTM-based peptide classifier
│   ├── train_classifier.py           # Classifier training script
│   ├── calibrate_temperature.py      # Temperature scaling calibration
│   ├── train_rl_ppo.py              # Basic PPO implementation
│   ├── rl_enhanced_env.py           # Enhanced environment with calibration
│   ├── ppo_with_kl.py               # PPO with KL penalty
│   ├── active_selection/bo_selector.py  # Bayesian optimization selector
│   └── evaluate_candidates_gpu.py   # GPU-accelerated evaluation
├── baseline_L1_classic_opt/         # Genetic algorithm baseline
├── active_loop_round.py             # Active learning orchestration
├── run_pipeline.py                  # One-click pipeline script
└── config.yaml                      # Configuration file
```

## Configuration

Key configuration sections:

### RL Settings
```yaml
rl:
  learning_rate: 0.0003
  gamma: 0.99
  clip_epsilon: 0.2
  entropy_coef: 0.01
```

### Enhanced Features
```yaml
rl_enhanced:
  kl_penalty:
    enabled: false
    beta: 0.01
    target_kl: 0.01
    adaptive: true

active_selection:
  ucb:
    enabled: true
    kappa: 1.0
    diversity_weight: 0.1
    top_k: 10
```

### Reward Weights
```yaml
reward_weights:
  target_prob: 1.0      # Calibrated classifier probability
  specificity: 0.5      # Multi-receptor specificity
  physchem: 0.3         # Physicochemical constraints
  motif: 0.2            # Known BBB penetration motifs
  safety: 0.1           # Safety considerations
```

## Usage Examples

### Running Active Learning
```bash
# Round 1: Initial candidate generation
python active_loop_round.py --config config.yaml --target TFRC --round 1

# After experimental validation, add new data to data/new_experimental_round_1.csv
# Then run round 2 with updated model
python active_loop_round.py --config config.yaml --target TFRC --round 2
```

### Using KL Penalty
```python
from src.ppo_with_kl import PPOWithKL, create_natural_policy_from_data

# Create natural policy from training data
natural_policy = create_natural_policy_from_data(config, device)

# Initialize PPO with KL penalty
ppo = PPOWithKL(model, kl_config=config["rl_enhanced"]["kl_penalty"])
ppo.set_reference_policy(natural_policy)
```

### Bayesian Optimization Selection
```python
from src.active_selection.bo_selector import BOSelector

selector = BOSelector(config)
selected_candidates = selector.select_candidates(
    "runs/ppo/candidates.csv",
    "results/selected_candidates.csv"
)
```

## Output Files

- `checkpoints/classifier_best_calibrated.pth`: Calibrated classifier
- `results/temperature.json`: Temperature scaling parameters
- `runs/ppo/top_sequences_epoch_*.csv`: RL-generated candidates
- `results/active_learning/selected_round_*.csv`: BO-selected candidates
- `results/saliency/*.csv`: Positional importance analysis

## Requirements

See `requirements.txt` for full dependency list.

Key dependencies:
- PyTorch >= 1.9.0
- NumPy, Pandas
- PyYAML
- Scikit-learn

## Citation

If you use this code in your research, please cite:

```bibtex
@software{bbb_peptide_rl,
  title = {BBB-Penetrating Peptide Design with Enhanced RL Pipeline},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/your-repo/bbb-peptide-rl}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
