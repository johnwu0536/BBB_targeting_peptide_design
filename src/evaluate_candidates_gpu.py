#!/usr/bin/env python3
"""
GPU-based evaluation of generated peptide candidates using calibrated classifier.
"""

import os
import sys
import argparse
import yaml
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.models.classifier import PeptideClassifier
from src.utils.encoding import clean_sequence, validate_sequence_length

# Set up logging
logging.basicConfig(level=logging.INFO, format='[evaluate_candidates_gpu] %(message)s')
logger = logging.getLogger(__name__)


def load_calibrated_classifier(config: Dict, device: torch.device) -> PeptideClassifier:
    """Load calibrated classifier model."""
    # Try calibrated model first
    model_path = "checkpoints/classifier_best_calibrated.pth"
    if not os.path.exists(model_path):
        model_path = config["paths"]["classifier_ckpt"]
        if not os.path.exists(model_path):
            model_path = "checkpoints/classifier_best.pth"
    
    # Load temperature parameter
    temp_path = config["paths"]["temperature_json"]
    if os.path.exists(temp_path):
        import json
        with open(temp_path, 'r') as f:
            temp_data = json.load(f)
        temperature = temp_data.get("temperature", 1.0)
    else:
        temperature = 1.0
    
    logger.info(f"Loading model from {model_path} with temperature {temperature}")
    
    # Use the compatible loading method to handle config structure changes
    try:
        # First try the standard load method
        model = PeptideClassifier.load(model_path, config, device)
    except (KeyError, ValueError) as e:
        logger.warning(f"Standard loading failed: {e}. Using compatible loading method.")
        # Fall back to compatible loading method
        model = PeptideClassifier.load_compatible(model_path, device)
    
    model.eval()
    
    return model, temperature


def encode_sequence(sequence: str, max_length: int = 20) -> torch.Tensor:
    """Encode sequence as amino acid indices."""
    aa_alphabet = "ACDEFGHIKLMNPQRSTVWY"
    encoded = []
    
    for aa in sequence:
        if aa in aa_alphabet:
            encoded.append(aa_alphabet.index(aa) + 1)  # 1-indexed
        else:
            encoded.append(0)  # Unknown/padding
    
    # Pad to max_length
    if len(encoded) < max_length:
        encoded = encoded + [0] * (max_length - len(encoded))
    elif len(encoded) > max_length:
        encoded = encoded[:max_length]
    
    return torch.tensor(encoded, dtype=torch.long)


def evaluate_sequences(sequences: List[str], model: PeptideClassifier, 
                      temperature: float, device: torch.device, 
                      config: Dict) -> pd.DataFrame:
    """Evaluate sequences using the calibrated classifier."""
    results = []
    batch_size = 32
    
    # Create progress bar for evaluation
    total_batches = (len(sequences) + batch_size - 1) // batch_size
    pbar = tqdm(total=total_batches, desc="Evaluating sequences", unit="batch")
    
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i + batch_size]
        batch_encoded = []
        valid_indices = []
        
        # Encode valid sequences
        for j, seq in enumerate(batch_seqs):
            cleaned = clean_sequence(seq, strict=False)
            if cleaned and validate_sequence_length(cleaned, 
                                                  config["task"]["min_len"], 
                                                  config["task"]["max_len"]):
                encoded = encode_sequence(cleaned, config["task"]["max_len"])
                batch_encoded.append(encoded)
                valid_indices.append(j)
        
        if not batch_encoded:
            pbar.update(1)
            continue
            
        # Batch inference
        batch_tensor = torch.stack(batch_encoded).to(device)
        
        with torch.no_grad():
            logits = model(batch_tensor)
            # Apply temperature scaling
            calibrated_logits = logits / temperature
            probs = torch.softmax(calibrated_logits, dim=-1)
            
            # Get probabilities for positive class (class 1)
            positive_probs = probs[:, 1].cpu().numpy()
        
        # Store results
        for idx, prob in zip(valid_indices, positive_probs):
            original_seq = batch_seqs[idx]
            cleaned_seq = clean_sequence(original_seq, strict=False)
            results.append({
                "original_sequence": original_seq,
                "cleaned_sequence": cleaned_seq,
                "probability": prob,
                "length": len(cleaned_seq) if cleaned_seq else 0
            })
        
        # Update progress bar
        pbar.update(1)
        pbar.set_postfix({
            "evaluated": len(results),
            "current_batch": len(batch_encoded)
        })
    
    pbar.close()
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Evaluate peptide candidates using GPU")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file with sequences")
    parser.add_argument("--out", type=str, required=True, help="Output CSV file")
    parser.add_argument("--sequence_col", type=str, default="sequence", help="Column name for sequences")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Device setup - prefer GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info("Loading calibrated classifier...")
    model, temperature = load_calibrated_classifier(config, device)
    
    # Load sequences
    logger.info(f"Loading sequences from {args.input}")
    try:
        df_input = pd.read_csv(args.input)
        if args.sequence_col not in df_input.columns:
            logger.error(f"Column '{args.sequence_col}' not found in input file")
            logger.info(f"Available columns: {list(df_input.columns)}")
            sys.exit(1)
        
        sequences = df_input[args.sequence_col].dropna().tolist()
        logger.info(f"Loaded {len(sequences)} sequences")
        
    except Exception as e:
        logger.error(f"Error loading input file: {e}")
        sys.exit(1)
    
    # Evaluate sequences
    logger.info("Evaluating sequences...")
    results_df = evaluate_sequences(sequences, model, temperature, device, config)
    
    # Sort by probability (descending)
    results_df = results_df.sort_values("probability", ascending=False)
    
    # Add rank
    results_df["rank"] = range(1, len(results_df) + 1)
    
    # Save results
    logger.info(f"Saving results to {args.out}")
    results_df.to_csv(args.out, index=False)
    
    # Print summary
    logger.info(f"Evaluation completed:")
    logger.info(f"  - Total sequences evaluated: {len(results_df)}")
    logger.info(f"  - Top probability: {results_df['probability'].max():.4f}")
    logger.info(f"  - Mean probability: {results_df['probability'].mean():.4f}")
    logger.info(f"  - Median probability: {results_df['probability'].median():.4f}")
    
    # Show top 5 sequences
    logger.info("Top 5 sequences:")
    for i, row in results_df.head().iterrows():
        logger.info(f"  {row['rank']}. {row['cleaned_sequence']} (p={row['probability']:.4f})")


if __name__ == "__main__":
    main()
