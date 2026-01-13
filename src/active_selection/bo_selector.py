#!/usr/bin/env python3
"""
Bayesian Optimization selector for candidate peptide selection.
Implements UCB acquisition function with uncertainty estimation for active learning.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.classifier import PeptideClassifier
from src.utils.encoding import clean_sequence

# Set up logging
logging.basicConfig(level=logging.INFO, format='[bo_selector] %(message)s')
logger = logging.getLogger(__name__)


class BOSelector:
    """
    Bayesian Optimization selector for peptide candidate selection.
    
    Uses uncertainty estimation and acquisition functions to select
    the most promising candidates for experimental validation.
    """
    
    def __init__(self, config: Dict, device: str = "cpu"):
        self.config = config
        self.device = device
        
        # BO configuration
        self.bo_config = config.get("active_selection", {}).get("ucb", {
            "enabled": True,
            "kappa": 1.0,
            "diversity_weight": 0.1,
            "top_k": 10
        })
        
        # Load classifier for uncertainty estimation
        self.classifier = None
        self._load_classifier()
        
        logger.info(f"BO Selector initialized: kappa={self.bo_config['kappa']}, "
                   f"diversity_weight={self.bo_config['diversity_weight']}")
    
    def _load_classifier(self):
        """Load classifier model for uncertainty estimation."""
        try:
            model_path = self.config["paths"]["classifier_ckpt"]
            if not os.path.exists(model_path):
                model_path = "checkpoints/classifier_best.pth"
            
            self.classifier = PeptideClassifier.load(model_path, self.config, self.device)
            self.classifier.eval()
            logger.info("Classifier loaded for uncertainty estimation")
        except Exception as e:
            logger.warning(f"Could not load classifier for uncertainty estimation: {e}")
            self.classifier = None
    
    def estimate_uncertainty(self, sequences: List[str]) -> np.ndarray:
        """
        Estimate uncertainty for candidate sequences.
        
        Args:
            sequences: List of peptide sequences
            
        Returns:
            Array of uncertainty estimates
        """
        if self.classifier is None:
            # Return uniform uncertainty if no classifier available
            return np.ones(len(sequences)) * 0.5
        
        uncertainties = []
        
        for seq in sequences:
            try:
                # Clean sequence
                cleaned = clean_sequence(seq, strict=False)
                if not cleaned:
                    uncertainties.append(0.5)  # Default uncertainty for invalid sequences
                    continue
                
                # Encode sequence
                encoded = self._encode_sequence(cleaned)
                encoded_tensor = torch.tensor(encoded).unsqueeze(0).to(self.device)
                
                # Use MC Dropout for uncertainty estimation
                uncertainty = self._mc_dropout_uncertainty(encoded_tensor)
                uncertainties.append(uncertainty)
                
            except Exception as e:
                logger.warning(f"Error estimating uncertainty for sequence {seq}: {e}")
                uncertainties.append(0.5)
        
        return np.array(uncertainties)
    
    def _mc_dropout_uncertainty(self, encoded_tensor: torch.Tensor, n_samples: int = 10) -> float:
        """
        Estimate uncertainty using Monte Carlo Dropout.
        
        Args:
            encoded_tensor: Encoded sequence tensor
            n_samples: Number of MC samples
            
        Returns:
            Uncertainty estimate (standard deviation of predictions)
        """
        if self.classifier is None:
            return 0.5
        
        # Enable dropout for uncertainty estimation
        self.classifier.train()
        
        predictions = []
        
        for _ in range(n_samples):
            with torch.no_grad():
                logits = self.classifier(encoded_tensor)
                probs = torch.softmax(logits, dim=1)
                positive_prob = probs[0, 1].item()
                predictions.append(positive_prob)
        
        # Disable dropout after uncertainty estimation
        self.classifier.eval()
        
        # Uncertainty is standard deviation of predictions
        uncertainty = np.std(predictions)
        return uncertainty
    
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
    
    def compute_acquisition_scores(self, sequences: List[str], 
                                 probabilities: np.ndarray,
                                 uncertainties: np.ndarray) -> np.ndarray:
        """
        Compute acquisition scores using UCB (Upper Confidence Bound).
        
        Args:
            sequences: List of peptide sequences
            probabilities: Predicted probabilities
            uncertainties: Uncertainty estimates
            
        Returns:
            Array of acquisition scores
        """
        kappa = self.bo_config["kappa"]
        
        # UCB acquisition function: mean + kappa * uncertainty
        acquisition_scores = probabilities + kappa * uncertainties
        
        # Apply diversity bonus
        if self.bo_config.get("diversity_weight", 0) > 0:
            diversity_bonus = self._compute_diversity_bonus(sequences)
            acquisition_scores += self.bo_config["diversity_weight"] * diversity_bonus
        
        return acquisition_scores
    
    def _compute_diversity_bonus(self, sequences: List[str]) -> np.ndarray:
        """
        Compute diversity bonus to encourage exploration of different regions.
        
        Args:
            sequences: List of peptide sequences
            
        Returns:
            Array of diversity bonuses
        """
        if len(sequences) <= 1:
            return np.zeros(len(sequences))
        
        # Simple diversity measure based on sequence similarity
        diversity_bonuses = np.zeros(len(sequences))
        
        for i, seq1 in enumerate(sequences):
            similarities = []
            for j, seq2 in enumerate(sequences):
                if i != j:
                    similarity = self._sequence_similarity(seq1, seq2)
                    similarities.append(similarity)
            
            if similarities:
                # Bonus inversely proportional to average similarity
                avg_similarity = np.mean(similarities)
                diversity_bonuses[i] = 1.0 - avg_similarity
        
        return diversity_bonuses
    
    def _sequence_similarity(self, seq1: str, seq2: str) -> float:
        """
        Compute simple sequence similarity.
        
        Args:
            seq1: First sequence
            seq2: Second sequence
            
        Returns:
            Similarity score between 0 and 1
        """
        # Simple character-based similarity
        if not seq1 or not seq2:
            return 0.0
        
        # Use Jaccard similarity of character sets
        set1 = set(seq1)
        set2 = set(seq2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def select_candidates(self, candidate_file: str, output_file: str) -> pd.DataFrame:
        """
        Select top candidates using Bayesian Optimization.
        
        Args:
            candidate_file: Path to CSV file with candidate sequences and scores
            output_file: Path to save selected candidates
            
        Returns:
            DataFrame with selected candidates
        """
        logger.info(f"Selecting candidates from {candidate_file}")
        
        # Load candidate data
        try:
            candidates_df = pd.read_csv(candidate_file)
        except Exception as e:
            logger.error(f"Error loading candidate file {candidate_file}: {e}")
            return pd.DataFrame()
        
        if candidates_df.empty:
            logger.warning("No candidates found in input file")
            return pd.DataFrame()
        
        # Extract sequences and probabilities
        sequences = candidates_df['sequence'].tolist()
        
        # Use reward column if available, otherwise use default probabilities
        if 'reward' in candidates_df.columns:
            probabilities = candidates_df['reward'].values
        elif 'probability' in candidates_df.columns:
            probabilities = candidates_df['probability'].values
        else:
            # If no probability column, use uniform probabilities
            probabilities = np.ones(len(sequences)) * 0.5
            logger.warning("No probability/reward column found, using uniform probabilities")
        
        # Estimate uncertainties
        uncertainties = self.estimate_uncertainty(sequences)
        
        # Compute acquisition scores
        acquisition_scores = self.compute_acquisition_scores(sequences, probabilities, uncertainties)
        
        # Add scores to dataframe
        candidates_df['uncertainty'] = uncertainties
        candidates_df['acquisition_score'] = acquisition_scores
        
        # Select top candidates
        top_k = self.bo_config.get("top_k", 10)
        selected_indices = np.argsort(acquisition_scores)[-top_k:][::-1]
        selected_candidates = candidates_df.iloc[selected_indices].copy()
        
        # Add selection metadata
        selected_candidates['selection_round'] = self._get_selection_round(output_file)
        selected_candidates['selection_method'] = 'BO_UCB'
        
        # Save selected candidates
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        selected_candidates.to_csv(output_file, index=False)
        
        logger.info(f"Selected {len(selected_candidates)} candidates, saved to {output_file}")
        
        # Log selection statistics
        self._log_selection_stats(selected_candidates, probabilities, uncertainties)
        
        return selected_candidates
    
    def _get_selection_round(self, output_file: str) -> int:
        """Extract selection round from output filename."""
        import re
        match = re.search(r'round(\d+)', output_file)
        if match:
            return int(match.group(1))
        return 1
    
    def _log_selection_stats(self, selected_candidates: pd.DataFrame,
                           probabilities: np.ndarray, uncertainties: np.ndarray):
        """Log selection statistics."""
        if selected_candidates.empty:
            return
        
        avg_prob = selected_candidates['acquisition_score'].mean()
        avg_uncertainty = selected_candidates['uncertainty'].mean()
        min_prob = selected_candidates['acquisition_score'].min()
        max_prob = selected_candidates['acquisition_score'].max()
        
        logger.info(f"Selection stats - Avg score: {avg_prob:.3f}, "
                   f"Avg uncertainty: {avg_uncertainty:.3f}, "
                   f"Range: [{min_prob:.3f}, {max_prob:.3f}]")


def main():
    """Command-line interface for BO selector."""
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="Bayesian Optimization candidate selector")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--input", type=str, required=True, help="Input candidate CSV file")
    parser.add_argument("--output", type=str, required=True, help="Output selected candidates CSV file")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create selector and run selection
    selector = BOSelector(config)
    selected_candidates = selector.select_candidates(args.input, args.output)
    
    if selected_candidates.empty:
        logger.error("No candidates selected")
        return 1
    
    logger.info(f"Successfully selected {len(selected_candidates)} candidates")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
