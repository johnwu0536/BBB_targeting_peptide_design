#!/usr/bin/env python3
"""
Enhanced PPO implementation with KL penalty for policy regularization.
Prevents policy drift from natural peptide distribution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='[ppo_with_kl] %(message)s')
logger = logging.getLogger(__name__)


class PPOWithKL:
    """
    Proximal Policy Optimization with KL penalty for policy regularization.
    
    The KL penalty constrains the policy from drifting too far from a reference
    distribution, improving sample efficiency and preventing mode collapse.
    """
    
    def __init__(self, model: nn.Module, lr: float = 1e-4, gamma: float = 0.99,
                 epsilon: float = 0.2, value_coef: float = 0.5, entropy_coef: float = 0.01,
                 kl_config: Optional[Dict] = None):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # KL penalty configuration
        self.kl_config = kl_config or {
            "enabled": False,
            "beta": 0.01,
            "target_kl": 0.01,
            "adaptive": True
        }
        
        # Reference policy for KL calculation
        self.reference_policy = None
        self.kl_enabled = self.kl_config["enabled"]
        
        if self.kl_enabled:
            logger.info(f"KL penalty enabled: beta={self.kl_config['beta']}, "
                       f"target_kl={self.kl_config['target_kl']}, "
                       f"adaptive={self.kl_config['adaptive']}")
    
    def set_reference_policy(self, reference_policy: nn.Module):
        """Set reference policy for KL divergence calculation."""
        self.reference_policy = reference_policy
        if self.reference_policy:
            logger.info("Reference policy set for KL regularization")
    
    def compute_kl_divergence(self, states: torch.Tensor, current_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between current policy and reference policy.
        
        Args:
            states: Input states
            current_logits: Current policy logits
            
        Returns:
            KL divergence tensor
        """
        if not self.kl_enabled or self.reference_policy is None:
            return torch.tensor(0.0, device=states.device)
        
        with torch.no_grad():
            # Get reference policy logits
            ref_logits, _ = self.reference_policy(states)
            
            # Create distributions
            current_dist = torch.distributions.Categorical(logits=current_logits)
            ref_dist = torch.distributions.Categorical(logits=ref_logits)
            
            # Compute KL divergence
            kl_div = torch.distributions.kl_divergence(current_dist, ref_dist)
            
            return kl_div.mean()
    
    def update(self, states, actions, old_log_probs, returns, advantages):
        """
        Update policy using PPO with optional KL penalty.
        
        Args:
            states: Input states
            actions: Actions taken
            old_log_probs: Log probabilities of actions under old policy
            returns: Returns (discounted cumulative rewards)
            advantages: Advantage estimates
            
        Returns:
            Dictionary with loss components and metrics
        """
        # Get current policy
        action_logits, values = self.model(states)
        dist = torch.distributions.Categorical(logits=action_logits)
        
        # Compute new log probabilities and entropy
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        # PPO ratio
        ratio = (new_log_probs - old_log_probs).exp()
        
        # PPO objectives
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        
        # Loss components
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = 0.5 * (returns - values).pow(2).mean()
        entropy_loss = -entropy
        
        # KL penalty
        kl_div = torch.tensor(0.0, device=states.device)
        kl_penalty = torch.tensor(0.0, device=states.device)
        
        if self.kl_enabled:
            kl_div = self.compute_kl_divergence(states, action_logits)
            kl_penalty = self.kl_config["beta"] * kl_div
            
            # Adaptive KL penalty (optional)
            if self.kl_config.get("adaptive", False):
                target_kl = self.kl_config.get("target_kl", 0.01)
                if kl_div > 1.5 * target_kl:
                    # Increase beta if KL is too high
                    self.kl_config["beta"] *= 1.5
                elif kl_div < target_kl / 1.5:
                    # Decrease beta if KL is too low
                    self.kl_config["beta"] *= 0.5
        
        # Total loss
        loss = (policy_loss + 
                self.value_coef * value_loss + 
                self.entropy_coef * entropy_loss + 
                kl_penalty)
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Prepare metrics
        metrics = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "total_loss": loss.item(),
            "kl_divergence": kl_div.item() if self.kl_enabled else 0.0,
            "kl_penalty": kl_penalty.item() if self.kl_enabled else 0.0,
            "kl_beta": self.kl_config["beta"] if self.kl_enabled else 0.0
        }
        
        return metrics


class NaturalPeptidePolicy:
    """
    Simple policy that represents natural peptide distribution.
    Can be initialized from data or use uniform/biased priors.
    """
    
    def __init__(self, vocab_size: int, max_length: int, device: torch.device):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.device = device
        
        # Simple LSTM-based policy for natural peptides
        self.embedding = nn.Embedding(vocab_size, 64)
        self.lstm = nn.LSTM(64, 128, batch_first=True, bidirectional=True)
        self.output = nn.Linear(128 * 2, vocab_size)
        
        self.to(device)
    
    def forward(self, x: torch.Tensor):
        """Forward pass for natural peptide policy."""
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        last_hidden = lstm_out[:, -1, :]  # Use last hidden state
        logits = self.output(last_hidden)
        values = torch.zeros(x.size(0), device=x.device)  # Dummy values
        
        return logits, values
    
    def initialize_from_data(self, sequences: list):
        """
        Initialize policy from natural peptide sequences.
        This creates a prior distribution based on observed data.
        """
        if not sequences:
            logger.warning("No sequences provided for natural policy initialization")
            return
        
        # Count amino acid frequencies
        aa_counts = {}
        total_count = 0
        
        for seq in sequences:
            for aa in seq:
                aa_counts[aa] = aa_counts.get(aa, 0) + 1
                total_count += 1
        
        if total_count > 0:
            # Create bias based on natural frequencies
            aa_alphabet = "ACDEFGHIKLMNPQRSTVWY"
            biases = []
            
            for aa in aa_alphabet:
                freq = aa_counts.get(aa, 0) / total_count
                # Convert frequency to logit bias
                bias = np.log(freq + 1e-8)  # Add small epsilon to avoid log(0)
                biases.append(bias)
            
            # Apply bias to output layer
            bias_tensor = torch.tensor(biases, dtype=torch.float32, device=self.device)
            self.output.bias.data = bias_tensor
            
            logger.info(f"Natural peptide policy initialized from {len(sequences)} sequences")
        else:
            logger.warning("No valid sequences found for natural policy initialization")


def create_natural_policy_from_data(config: Dict, device: torch.device, 
                                  data_file: str = "data/Binding_peptide_sequence.csv") -> NaturalPeptidePolicy:
    """
    Create natural peptide policy from training data.
    
    Args:
        config: Configuration dictionary
        device: Device to place model on
        data_file: Path to peptide data file
        
    Returns:
        Natural peptide policy model
    """
    import pandas as pd
    
    # Load natural peptide sequences
    try:
        df = pd.read_csv(data_file)
        if 'sequence' in df.columns:
            sequences = df['sequence'].dropna().tolist()
            logger.info(f"Loaded {len(sequences)} natural peptide sequences")
        else:
            sequences = []
            logger.warning(f"No 'sequence' column found in {data_file}")
    except Exception as e:
        logger.warning(f"Could not load natural peptide data: {e}")
        sequences = []
    
    # Create natural policy
    vocab_size = 21  # 20 AA + padding
    max_length = config["task"]["max_len"]
    natural_policy = NaturalPeptidePolicy(vocab_size, max_length, device)
    
    # Initialize from data if available
    if sequences:
        natural_policy.initialize_from_data(sequences)
    
    return natural_policy
