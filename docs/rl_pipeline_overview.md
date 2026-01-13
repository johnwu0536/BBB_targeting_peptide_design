# RL Pipeline Overview for BBB-Penetrating Peptide Design

## Current System Architecture

### 1. Classifier Training and Calibration
- **Training Script**: `src/train_classifier.py`
- **Model**: `src/models/classifier.py` - LSTM-based classifier with attention mechanism
- **Calibration**: `src/calibrate_temperature.py` - Temperature scaling for probability calibration
- **Checkpoints**: 
  - `checkpoints/classifier_best.pth` - Original trained model
  - `checkpoints/classifier_best_calibrated.pth` - Calibrated model
- **Temperature Data**: `results/temperature.json` - Contains calibrated temperature parameter

### 2. PPO / RL Core Implementation
- **Main Script**: `src/train_rl_ppo.py` - Complete PPO implementation
- **Environment**: `PeptideEnvironment` class in `src/train_rl_ppo.py`
- **Policy Network**: `ActorCritic` class with LSTM backbone
- **PPO Algorithm**: `PPO` class implementing proximal policy optimization

### 3. Candidate Evaluation
- **GPU Evaluation**: `src/evaluate_candidates_gpu.py` - Batch evaluation using calibrated classifier
- **Positional Saliency**: `src/explain/positional_saliency.py` - Leave-one-out importance analysis

### 4. Baselines
- **L1 Baseline**: `baseline_L1_classic_opt/` - Genetic algorithm optimization without AI
- **Scoring Functions**: `baseline_L1_classic_opt/src/score.py` - Hand-crafted physicochemical scoring

## Current Reward Definition

### Location
- **File**: `src/train_rl_ppo.py`
- **Function**: `PeptideEnvironment._compute_final_reward()`

### Reward Components
```python
rewards = {
    "target_prob": classifier_probability,  # From classifier prediction
    "physchem": simplified_physchem_score,  # Simplified physicochemical constraints
    "motif": motif_bonus,                   # Placeholder motif bonus
    "safety": safety_penalty,               # Placeholder safety score
    "specificity": 0.5                      # Placeholder specificity
}
```

### Weighted Sum
```python
total_reward = sum(weight * rewards[component] for component, weight in reward_weights.items())
```

### Current Weights (from config.yaml)
```yaml
reward_weights:
  target_prob: 1.0
  specificity: 0.5
  physchem: 0.3
  motif: 0.2
  safety: 0.1
```

## Classifier Usage in RL

### How Classifier is Invoked
- **Location**: `PeptideEnvironment._compute_final_reward()` method
- **Process**:
  1. Sequence is encoded using `_encode_sequence()`
  2. Encoded tensor is moved to classifier device
  3. Classifier predicts probabilities: `probs = self.classifier(encoded)`
  4. Positive class probability is extracted: `target_prob = probs[0, 1].item()`

### Current Limitations
1. **No Temperature Calibration**: RL uses raw classifier probabilities, not calibrated ones
2. **Simplified Physchem**: Current RL physchem scoring is simplified compared to L1 baseline
3. **No KL Regularization**: Policy can drift without constraints
4. **No Active Learning**: No BO-based candidate selection
5. **No Confidence Thresholding**: Success defined by p>0.5 only

## Data Flow

1. **Training**: `train_classifier.py` → `classifier_best.pth`
2. **Calibration**: `calibrate_temperature.py` → `temperature.json` + `classifier_best_calibrated.pth`
3. **RL Training**: `train_rl_ppo.py` → `runs/ppo/top_sequences_epoch_*.csv`
4. **Evaluation**: `evaluate_candidates_gpu.py` → `results/eval_gpu.csv`
5. **Analysis**: `positional_saliency.py` → `results/saliency/`

## Key Configuration Parameters

### Classifier
```yaml
classifier:
  embedding: 128
  hidden: 256
  dropout: 0.3
  batch_size: 32
```

### RL
```yaml
rl:
  learning_rate: 0.0003
  gamma: 0.99
  clip_epsilon: 0.2
  entropy_coef: 0.01
```

### Task Constraints
```yaml
task:
  min_len: 8
  max_len: 20
```

## Areas for Improvement (Identified)

1. **Surrogate-RL Loop**: RL should use calibrated classifier probabilities
2. **KL Penalty**: Add policy regularization against natural peptide distribution
3. **BO + RL Hybrid**: Implement Bayesian optimization for candidate selection
4. **Confidence Thresholding**: Redefine success using high-confidence probabilities
5. **Better Physchem Integration**: Use L1 baseline scoring functions in RL
