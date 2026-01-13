#!/usr/bin/env python3
"""
Enhanced RL environment with calibrated classifier probabilities,
better physicochemical scoring, and KL regularization support.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
import logging
from typing import Dict, List, Tuple

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.encoding import clean_sequence, validate_sequence_length
from src.models.classifier import PeptideClassifier

# Set up logging
logging.basicConfig(level=logging.INFO, format='[rl_enhanced_env] %(message)s')
logger = logging.getLogger(__name__)


class CalibratedClassifierWrapper:
    """Wrapper for calibrated classifier with temperature scaling."""
    
    def __init__(self, config: Dict, device: torch.device):
        self.config = config
        self.device = device
        
        # Load calibrated model
        calibrated_path = "checkpoints/classifier_best_calibrated.pth"
        if not os.path.exists(calibrated_path):
            calibrated_path = config["paths"]["classifier_ckpt"]
        
        logger.info(f"Loading calibrated classifier from {calibrated_path}")
        self.model = PeptideClassifier.load(calibrated_path, config, device)
        self.model.eval()
        
        # Load temperature parameter
        temp_path = config["paths"]["temperature_json"]
        if os.path.exists(temp_path):
            with open(temp_path, 'r') as f:
                temp_data = json.load(f)
            self.temperature = temp_data.get("temperature", 1.0)
        else:
            self.temperature = 1.0
        
        logger.info(f"Using temperature scaling: {self.temperature:.4f}")
    
    def predict_calibrated_prob(self, sequence: str) -> float:
        """Predict calibrated probability for a sequence."""
        if not sequence:
            return 0.0
        
        # Clean sequence
        cleaned = clean_sequence(sequence, strict=False)
        if not cleaned:
            return 0.0
        
        # Encode sequence
        encoded = self._encode_sequence(cleaned)
        encoded_tensor = torch.tensor(encoded).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(encoded_tensor)
            # Apply temperature scaling
            scaled_logits = logits / self.temperature
            probs = torch.softmax(scaled_logits, dim=1)
            calibrated_prob = probs[0, 1].item()  # Positive class probability
        
        return calibrated_prob
    
    def _encode_sequence(self, sequence: str) -> List[int]:
        """Encode sequence as amino acid indices."""
        aa_alphabet = "ACDEFGHIKLMNPQRSTVWY"
        encoded = []
        
        for aa in sequence:
            if aa in aa_alphabet:
                encoded.append(aa_alphabet.index(aa) + 1)  # 1-indexed
            else:
                encoded.append(0)  # Unknown/padding
        
        # Pad to max_length
        max_len = self.config["task"]["max_len"]
        if len(encoded) < max_len:
            encoded = encoded + [0] * (max_len - len(encoded))
        
        return encoded


class EnhancedPeptideEnvironment:
    """Enhanced environment with calibrated classifier and better scoring."""
    
    def __init__(self, config: Dict, target_receptor: str, device: torch.device):
        self.config = config
        self.target_receptor = target_receptor
        self.device = device
        
        # Environment parameters
        self.min_len = config["task"]["min_len"]
        self.max_len = config["task"]["max_len"]
        self.vocab_size = 21  # 20 AA + padding
        self.aa_alphabet = "ACDEFGHIKLMNPQRSTVWY"
        
        # Reward weights
        self.reward_weights = config["reward_weights"]
        self.physchem_constraints = config["physchem_constraints"]
        
        # Load calibrated classifier
        self.classifier = CalibratedClassifierWrapper(config, device)
        
        # Load L1 baseline scoring functions if available
        try:
            sys.path.append(str(Path(__file__).parent.parent / "baseline_L1_classic_opt" / "src"))
            from score import score_sequence, get_detailed_scores
            from aa import calculate_hydropathy, calculate_charge, count_cysteines, count_repeats, contains_motif
            self._has_l1_scoring = True
            self._score_sequence = score_sequence
            self._get_detailed_scores = get_detailed_scores
            logger.info("L1 baseline scoring functions loaded successfully")
        except ImportError:
            self._has_l1_scoring = False
            logger.warning("L1 baseline scoring functions not available, using simplified scoring")
        
        # Current state
        self.current_sequence = []
        self.done = False
        
        # Confidence thresholds
        self.thresholds = config.get("classifier", {}).get("thresholds", {
            "high": 0.8,
            "medium": 0.5
        })
    
    def reset(self) -> torch.Tensor:
        """Reset environment to initial state."""
        self.current_sequence = []
        self.done = False
        return self._get_state()
    
    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, Dict]:
        """Take action and return next state, reward, done, info."""
        if self.done:
            raise ValueError("Environment is done, call reset() first")
        
        # Add amino acid to sequence
        if action < len(self.aa_alphabet):
            aa = self.aa_alphabet[action]
            self.current_sequence.append(aa)
        
        # Check if sequence is complete
        sequence = ''.join(self.current_sequence)
        if len(sequence) >= self.max_len or action == len(self.aa_alphabet):  # Stop token
            self.done = True
            reward = self._compute_enhanced_reward(sequence)
        else:
            reward = 0.0  # Intermediate step reward
        
        next_state = self._get_state()
        info = {"sequence": sequence, "length": len(sequence)}
        
        return next_state, reward, self.done, info
    
    def _get_state(self) -> torch.Tensor:
        """Get current state as tensor."""
        sequence = ''.join(self.current_sequence)
        encoded = self._encode_sequence(sequence)
        return torch.tensor(encoded, dtype=torch.long)
    
    def _encode_sequence(self, sequence: str) -> List[int]:
        """Encode sequence as amino acid indices."""
        encoded = []
        for aa in sequence:
            if aa in self.aa_alphabet:
                encoded.append(self.aa_alphabet.index(aa) + 1)  # 1-indexed
            else:
                encoded.append(0)  # Unknown/padding
        
        # Pad to max_length
        if len(encoded) < self.max_len:
            encoded = encoded + [0] * (self.max_len - len(encoded))
        
        return encoded
    
    def _compute_enhanced_reward(self, sequence: str) -> float:
        """Compute enhanced reward with calibrated probabilities and better scoring."""
        if len(sequence) < self.min_len:
            return -1.0  # Penalty for too short sequences
        
        # Clean sequence
        cleaned = clean_sequence(sequence, strict=False)
        if not cleaned:
            return -1.0
        
        # Compute reward components
        rewards = {}
        
        # 1. Calibrated target receptor probability with confidence bonus
        calibrated_prob = self.classifier.predict_calibrated_prob(cleaned)
        rewards["target_prob"] = self._apply_confidence_bonus(calibrated_prob)
        
        # 2. Enhanced physicochemical scoring
        if self._has_l1_scoring:
            # Use L1 baseline scoring functions
            physchem_score = self._score_sequence(
                cleaned, 
                self.physchem_constraints, 
                self.reward_weights
            )
            # Normalize to [0, 1] range
            rewards["physchem"] = max(0.0, min(1.0, physchem_score / 2.0))
        else:
            # Fallback to simplified scoring
            rewards["physchem"] = self._compute_simplified_physchem_reward(cleaned)
        
        # 3. Motif score
        rewards["motif"] = self._compute_motif_score(cleaned)
        
        # 4. Safety score
        rewards["safety"] = self._compute_safety_score(cleaned)
        
        # 5. Specificity (placeholder - would need multi-class classifier)
        rewards["specificity"] = 0.5
        
        # Weighted sum
        total_reward = 0.0
        for component, weight in self.reward_weights.items():
            if component in rewards:
                total_reward += weight * rewards[component]
        
        return total_reward
    
    def _apply_confidence_bonus(self, probability: float) -> float:
        """Apply confidence-based bonus to probability."""
        high_threshold = self.thresholds["high"]
        medium_threshold = self.thresholds["medium"]
        
        if probability >= high_threshold:
            # High confidence bonus
            return probability + 0.2 * (probability - high_threshold)
        elif probability >= medium_threshold:
            # Medium confidence - no bonus
            return probability
        else:
            # Low confidence - penalty
            return probability * 0.5
    
    def _compute_simplified_physchem_reward(self, sequence: str) -> float:
        """Simplified physicochemical scoring (fallback)."""
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
        
        # Repeat penalty (simplified)
        max_repeat = max(len(sequence) - len(set(sequence)), 0)
        if max_repeat <= self.physchem_constraints["max_repeats"]:
            score += 0.2
        else:
            score -= 0.1 * (max_repeat - self.physchem_constraints["max_repeats"])
        
        return min(max(score, 0.0), 1.0)
    
    def _compute_motif_score(self, sequence: str) -> float:
        """Compute motif score."""
        # Known BBB penetration motifs
        bbb_motifs = ["YPR", "THR", "TRP", "RGD", "NGR", "KPV"]
        motif_count = sum(1 for motif in bbb_motifs if motif in sequence)
        return min(motif_count * 0.2, 1.0)  # Cap at 1.0
    
    def _compute_safety_score(self, sequence: str) -> float:
        """Compute safety score."""
        score = 0.0
        
        # Penalize cysteines (potential for disulfide bonds)
        cysteine_count = sequence.count('C')
        if cysteine_count == 0:
            score += 0.3
        elif cysteine_count <= 1:
            score += 0.1
        else:
            score -= 0.1 * (cysteine_count - 1)
        
        # Penalize very hydrophobic sequences
        if self._has_l1_scoring:
            try:
                hydropathy = calculate_hydropathy(sequence)
                if -1.0 <= hydropathy <= 1.0:
                    score += 0.2
            except:
                pass
        
        return min(max(score, 0.0), 1.0)
