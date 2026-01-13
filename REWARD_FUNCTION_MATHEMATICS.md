# Reward Function Mathematical Expression

## Overview

The reward function in the RL peptide design system is a multi-component weighted sum that evaluates generated peptide sequences based on multiple criteria. The function is defined in `src/train_rl_ppo.py` and configured in `config.yaml`.

## Complete Mathematical Formulation

### Main Reward Function

The total reward \( R_{\text{total}} \) for a peptide sequence \( s \) is given by:

\[
R_{\text{total}}(s) = \sum_{i=1}^{5} w_i \cdot r_i(s)
\]

Where:
- \( w_i \) are the reward weights from configuration
- \( r_i(s) \) are the individual reward components

### Reward Components

#### 1. Target Receptor Probability (\( r_{\text{target}} \))

\[
r_{\text{target}}(s) = P_{\text{classifier}}(s \in \text{positive class})
\]

Where \( P_{\text{classifier}} \) is the probability output from the trained binary classifier for the target receptor.

#### 2. Specificity Score (\( r_{\text{specificity}} \))

\[
r_{\text{specificity}}(s) = \text{specificity\_score}(s)
\]

Currently implemented as a placeholder value of 0.5. In a full implementation, this would be:

\[
r_{\text{specificity}}(s) = 1 - \max_{r \neq \text{target}} P_{\text{classifier}}(s \in \text{positive class for receptor } r)
\]

#### 3. Physicochemical Constraints (\( r_{\text{physchem}} \))

\[
r_{\text{physchem}}(s) = \sum_{j=1}^{3} c_j(s)
\]

Where:
- \( c_1(s) = \begin{cases} 
  0.2 & \text{if } l_{\min} \leq |s| \leq l_{\max} \\
  0 & \text{otherwise}
\end{cases} \)
- \( c_2(s) = \begin{cases} 
  0.2 & \text{if } \text{cys\_count}(s) \leq C_{\max} \\
  -0.1 \cdot (\text{cys\_count}(s) - C_{\max}) & \text{otherwise}
\end{cases} \)
- \( c_3(s) = \begin{cases} 
  0.2 & \text{if } \text{max\_repeat}(s) \leq R_{\max} \\
  -0.1 \cdot (\text{max\_repeat}(s) - R_{\max}) & \text{otherwise}
\end{cases} \)

With bounds:
- \( l_{\min} = 8 \), \( l_{\max} = 20 \) (from config)
- \( C_{\max} = 2 \) (max cysteines)
- \( R_{\max} = 3 \) (max consecutive repeats)

#### 4. Motif Score (\( r_{\text{motif}} \))

\[
r_{\text{motif}}(s) = \begin{cases}
0.1 & \text{if } \exists m \in M \text{ such that } m \subseteq s \\
0 & \text{otherwise}
\end{cases}
\]

Where \( M = \{\text{"YPR"}, \text{"THR"}, \text{"TRP"}\} \) are known binding motifs.

#### 5. Safety Score (\( r_{\text{safety}} \))

\[
r_{\text{safety}}(s) = \begin{cases}
0.1 & \text{if } \text{cys\_count}(s) = 0 \\
0 & \text{otherwise}
\end{cases}
\]

## Configuration Parameters

From `config.yaml`:

```yaml
reward_weights:
  target_prob: 1.0      # w₁
  specificity: 0.5      # w₂  
  physchem: 0.3         # w₃
  motif: 0.2            # w₄
  safety: 0.1           # w₅

physchem_constraints:
  max_cysteines: 2      # C_max
  max_repeats: 3        # R_max
```

## Complete Expanded Form

Substituting all components:

\[
\begin{aligned}
R_{\text{total}}(s) = &\ 1.0 \cdot P_{\text{classifier}}(s) \\
&+ 0.5 \cdot \text{specificity\_score}(s) \\
&+ 0.3 \cdot \left[ c_1(s) + c_2(s) + c_3(s) \right] \\
&+ 0.2 \cdot \mathbb{I}_{\{\exists m \in M: m \subseteq s\}} \\
&+ 0.1 \cdot \mathbb{I}_{\{\text{cys\_count}(s) = 0\}}
\end{aligned}
\]

Where:
- \( \mathbb{I}_{\{condition\}} \) is the indicator function (1 if condition true, 0 otherwise)
- \( P_{\text{classifier}}(s) \in [0, 1] \) is the classifier probability
- \( \text{specificity\_score}(s) \in [0, 1] \) (currently 0.5 placeholder)
- \( c_1(s), c_2(s), c_3(s) \) are physicochemical constraint scores
- \( M = \{\text{"YPR"}, \text{"THR"}, \text{"TRP"}\} \) are target motifs

## Implementation Details

### In Code (`src/train_rl_ppo.py`):

```python
def _compute_final_reward(self, sequence: str) -> float:
    # Compute reward components
    rewards = {}
    
    # 1. Target receptor probability
    rewards["target_prob"] = classifier_probability
    
    # 2. Physicochemical constraints
    rewards["physchem"] = self._compute_physchem_reward(sequence)
    
    # 3. Motif score
    rewards["motif"] = 0.1 if any(motif in sequence for motif in ["YPR", "THR", "TRP"]) else 0.0
    
    # 4. Safety score  
    rewards["safety"] = 0.1 if "C" not in sequence else 0.0
    
    # 5. Specificity (placeholder)
    rewards["specificity"] = 0.5
    
    # Weighted sum
    total_reward = 0.0
    for component, weight in self.reward_weights.items():
        if component in rewards:
            total_reward += weight * rewards[component]
    
    return total_reward
```

### Physicochemical Sub-function:

```python
def _compute_physchem_reward(self, sequence: str) -> float:
    score = 0.0
    
    # Length constraint
    if self.min_len <= len(sequence) <= self.max_len:
        score += 0.2
    
    # Cysteine penalty
    cysteine_count = sequence.count('C')
    if cysteine_count <= self.physchem_constraints["max_cysteines"]:
        score += 0.2
    else:
        score -= 0.1 * (cysteine_count - self.physchem_constraints["max_cysteines"])
    
    # Repeat penalty
    max_repeat = max(len(sequence) - len(set(sequence)), 0)
    if max_repeat <= self.physchem_constraints["max_repeats"]:
        score += 0.2
    else:
        score -= 0.1 * (max_repeat - self.physchem_constraints["max_repeats"])
    
    return min(max(score, 0.0), 1.0)
```

## Range and Normalization

- **Total Reward Range**: \( R_{\text{total}} \in [-0.3, 2.1] \) theoretically
- **Practical Range**: Typically \( R_{\text{total}} \in [0, 1.5] \) for valid sequences
- **Normalization**: Physicochemical component is clamped to [0, 1]
- **Classifier Output**: Already normalized to [0, 1]

## Optimization Objective

The RL agent aims to maximize the expected cumulative reward:

\[
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \gamma^t R_{\text{total}}(s_t) \right]
\]

Where:
- \( \pi_\theta \) is the policy parameterized by \( \theta \)
- \( \tau \) is a trajectory of states and actions
- \( \gamma = 0.99 \) is the discount factor
- \( T \) is the episode horizon (sequence length)

This reward function provides a comprehensive evaluation of peptide sequences considering binding affinity, specificity, physicochemical properties, known motifs, and safety considerations.
