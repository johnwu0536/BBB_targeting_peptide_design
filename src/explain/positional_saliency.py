#!/usr/bin/env python3
"""
Positional saliency analysis for peptide sequences.
Computes per-residue importance using leave-one-out probability drop.
"""

import os
import sys
import yaml
import argparse
import pandas as pd
import torch
import numpy as np
from typing import List, Dict, Tuple
import json
import csv

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.classifier import PeptideClassifier
from src.utils.encoding import clean_sequence, encode_sequence, AA_ALPHABET


def setup_device(config: dict) -> str:
    """Setup device for analysis."""
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("[positional_saliency] CUDA not available, falling back to CPU")
        device = "cpu"
    
    print(f"[positional_saliency] Using device: {device}")
    return device


def load_model_and_temperature(config: dict, device: str) -> Tuple[PeptideClassifier, float]:
    """Load trained model and temperature parameter."""
    # Load model
    model_path = config["paths"]["classifier_ckpt"]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    print(f"[positional_saliency] Loading model from {model_path}")
    model = PeptideClassifier.load(model_path, config, device)
    
    # Load temperature parameter
    temperature_path = config["paths"]["temperature_json"]
    temperature = 1.0  # default
    
    if os.path.exists(temperature_path):
        try:
            with open(temperature_path, 'r') as f:
                temp_data = json.load(f)
                temperature = temp_data.get("temperature", 1.0)
            print(f"[positional_saliency] Using calibrated temperature: {temperature:.4f}")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[positional_saliency] Warning: Could not load temperature from {temperature_path}: {e}")
            print("[positional_saliency] Using default temperature: 1.0")
    else:
        print(f"[positional_saliency] Warning: Temperature file not found: {temperature_path}")
        print("[positional_saliency] Using default temperature: 1.0")
    
    return model, temperature


def load_candidates(input_file: str, limit: int = None) -> List[str]:
    """Load candidate sequences from input file."""
    print(f"[positional_saliency] Loading candidates from {input_file}")
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    candidates = []
    
    # Try different file formats
    if input_file.endswith('.csv'):
        try:
            df = pd.read_csv(input_file)
            if 'sequence' in df.columns:
                candidates = df['sequence'].tolist()
            elif 'Sequence' in df.columns:
                candidates = df['Sequence'].tolist()
            else:
                # Assume first column contains sequences
                candidates = df.iloc[:, 0].tolist()
        except Exception as e:
            print(f"[positional_saliency] Warning: Could not read CSV: {e}")
            # Fall back to text file reading
    
    # If CSV failed or it's a text file, read as text
    if not candidates:
        try:
            with open(input_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Extract sequence (first token)
                        parts = line.split()
                        if parts:
                            candidates.append(parts[0])
        except Exception as e:
            raise ValueError(f"Could not read input file {input_file}: {e}")
    
    # Clean sequences
    cleaned_candidates = []
    for seq in candidates:
        if pd.isna(seq):
            continue
        cleaned = clean_sequence(str(seq), strict=False)
        if cleaned:
            cleaned_candidates.append(cleaned)
    
    print(f"[positional_saliency] Loaded {len(cleaned_candidates)}/{len(candidates)} valid sequences")
    
    if limit:
        cleaned_candidates = cleaned_candidates[:limit]
        print(f"[positional_saliency] Limited to {limit} sequences")
    
    if not cleaned_candidates:
        raise ValueError("No valid sequences found after cleaning")
    
    return cleaned_candidates


def compute_positional_saliency(model: PeptideClassifier, sequence: str, 
                               temperature: float, device: str) -> Dict:
    """
    Compute positional saliency using leave-one-out method.
    
    Args:
        model: Trained classifier
        sequence: Input peptide sequence
        temperature: Temperature for probability calibration
        device: Device to use
    
    Returns:
        Dictionary with saliency results
    """
    model.eval()
    
    # Encode original sequence
    max_length = len(sequence)
    original_encoded = encode_sequence(sequence, max_length).unsqueeze(0).to(device)
    
    # Get base probability
    with torch.no_grad():
        logits = model(original_encoded)
        # Apply temperature scaling
        scaled_logits = logits / temperature
        base_probs = torch.softmax(scaled_logits, dim=1)
        base_prob = base_probs[0, 1].item()  # Positive class probability
    
    # Compute leave-one-out probabilities
    importance_scores = []
    
    for i in range(len(sequence)):
        # Create sequence with residue i removed
        masked_sequence = sequence[:i] + sequence[i+1:]
        
        if not masked_sequence:  # Skip if sequence becomes empty
            importance_scores.append(0.0)
            continue
        
        # Encode masked sequence
        masked_encoded = encode_sequence(masked_sequence, max_length).unsqueeze(0).to(device)
        
        # Get masked probability
        with torch.no_grad():
            masked_logits = model(masked_encoded)
            scaled_masked_logits = masked_logits / temperature
            masked_probs = torch.softmax(scaled_masked_logits, dim=1)
            masked_prob = masked_probs[0, 1].item()
        
        # Compute importance as probability drop
        delta_prob = max(0, base_prob - masked_prob)
        importance_scores.append(delta_prob)
    
    # Normalize importance scores
    max_importance = max(importance_scores) if importance_scores else 1.0
    if max_importance > 0:
        normalized_scores = [score / max_importance for score in importance_scores]
    else:
        normalized_scores = [0.0] * len(importance_scores)
    
    return {
        'sequence': sequence,
        'base_prob': base_prob,
        'importance_raw': importance_scores,
        'importance': normalized_scores
    }


def save_results(results: List[Dict], output_csv: str, target: str):
    """Save saliency results to CSV."""
    print(f"[positional_saliency] Saving results to {output_csv}")
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['rank', 'sequence', 'pos', 'aa', 'importance_raw', 'importance', 'p_target', 'target'])
        
        for rank, result in enumerate(results, 1):
            sequence = result['sequence']
            base_prob = result['base_prob']
            importance_raw = result['importance_raw']
            importance = result['importance']
            
            for pos, (aa, raw_imp, norm_imp) in enumerate(zip(sequence, importance_raw, importance)):
                writer.writerow([rank, sequence, pos + 1, aa, raw_imp, norm_imp, base_prob, target])


def create_plots(results: List[Dict], output_dir: str, target: str):
    """Create bar plots for saliency analysis (if matplotlib is available)."""
    try:
        # Set non-interactive backend to avoid GUI issues
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        print(f"[positional_saliency] Creating plots in {output_dir}")
        
        # Create plots for top sequences
        for i, result in enumerate(results[:5]):  # Plot top 5 sequences
            sequence = result['sequence']
            importance = result['importance']
            
            plt.figure(figsize=(12, 6))
            positions = list(range(1, len(sequence) + 1))
            amino_acids = list(sequence)
            
            # Create bar plot
            bars = plt.bar(positions, importance, color='skyblue', alpha=0.7)
            
            # Color bars by amino acid type
            colors = {
                'A': 'lightblue', 'C': 'orange', 'D': 'red', 'E': 'darkred', 
                'F': 'purple', 'G': 'green', 'H': 'pink', 'I': 'yellow', 
                'K': 'blue', 'L': 'yellowgreen', 'M': 'brown', 'N': 'lightgreen',
                'P': 'gray', 'Q': 'darkgreen', 'R': 'navy', 'S': 'lightcoral',
                'T': 'gold', 'V': 'olive', 'W': 'indigo', 'Y': 'teal'
            }
            
            for j, (bar, aa) in enumerate(zip(bars, amino_acids)):
                bar.set_color(colors.get(aa, 'gray'))
                # Add amino acid label
                plt.text(positions[j], importance[j] + 0.01, aa, 
                        ha='center', va='bottom', fontweight='bold')
            
            plt.xlabel('Position')
            plt.ylabel('Normalized Importance')
            plt.title(f'Positional Saliency: {sequence}\nTarget: {target}')
            plt.ylim(0, 1.1)
            plt.grid(True, alpha=0.3)
            
            # Save plot
            plot_path = os.path.join(output_dir, f"saliency_rank_{i+1}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"[positional_saliency] Saved plot: {plot_path}")
    
    except ImportError as e:
        print(f"[positional_saliency] Warning: matplotlib not available, skipping plots: {e}")
    except Exception as e:
        print(f"[positional_saliency] Warning: Could not create plots: {e}")


def main():
    parser = argparse.ArgumentParser(description="Positional saliency analysis")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--input", type=str, required=True, help="Input candidate sequences file")
    parser.add_argument("--target", type=str, required=True, help="Target receptor name")
    parser.add_argument("--limit", type=int, default=50, help="Limit number of sequences to analyze")
    parser.add_argument("--outdir", type=str, default="results/saliency", help="Output directory")
    args = parser.parse_args()
    
    # Load config
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"[positional_saliency] Error: Config file {args.config} not found")
        return 1
    
    # Setup device
    device = setup_device(config)
    
    try:
        # Load model and temperature
        model, temperature = load_model_and_temperature(config, device)
        
        # Load candidates
        candidates = load_candidates(args.input, args.limit)
        
        # Compute positional saliency for each candidate
        print("[positional_saliency] Computing positional saliency...")
        results = []
        
        for i, sequence in enumerate(candidates, 1):
            print(f"[positional_saliency] Analyzing sequence {i}/{len(candidates)}: {sequence}")
            
            try:
                result = compute_positional_saliency(model, sequence, temperature, device)
                results.append(result)
            except Exception as e:
                print(f"[positional_saliency] Warning: Failed to analyze sequence {sequence}: {e}")
                continue
        
        if not results:
            print("[positional_saliency] Error: No sequences successfully analyzed")
            return 1
        
        # Sort by base probability (descending)
        results.sort(key=lambda x: x['base_prob'], reverse=True)
        
        # Save results
        output_csv = os.path.join(args.outdir, f"positional_saliency_{args.target}.csv")
        save_results(results, output_csv, args.target)
        
        # Create plots
        create_plots(results, args.outdir, args.target)
        
        print(f"[positional_saliency] Analysis completed. Results saved to {output_csv}")
        return 0
        
    except Exception as e:
        print(f"[positional_saliency] Error during analysis: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
