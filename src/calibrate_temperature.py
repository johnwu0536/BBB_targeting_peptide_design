#!/usr/bin/env python3
"""
Temperature scaling calibration for classifier outputs.
Improves probability calibration for better uncertainty estimation.
"""

import os
import sys
import yaml
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.classifier import PeptideClassifier
from src.utils.data_loader import load_peptide_data, create_data_loaders


class TemperatureScaler(nn.Module):
    """Temperature scaling for probability calibration."""
    
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * temperature)
    
    def forward(self, logits):
        """Scale logits by temperature."""
        return logits / self.temperature
    
    def calibrate_probs(self, logits):
        """Calibrate probabilities using temperature scaling."""
        with torch.no_grad():
            scaled_logits = self.forward(logits)
            return torch.softmax(scaled_logits, dim=1)


def setup_device(config: dict) -> str:
    """Setup device for calibration."""
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("[calibrate_temperature] CUDA not available, falling back to CPU")
        device = "cpu"
    
    print(f"[calibrate_temperature] Using device: {device}")
    return device


def load_model_and_data(config: dict, device: str):
    """Load trained model and validation data."""
    # Load model
    model_path = config["paths"]["classifier_ckpt"]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    print(f"[calibrate_temperature] Loading model from {model_path}")
    
    # Use classifier factory for robust model loading
    from src.models.classifier_factory import load_classifier_from_checkpoint
    model = load_classifier_from_checkpoint(config, model_path, device)
    
    # Load validation data
    print("[calibrate_temperature] Loading validation data...")
    sequences, labels = load_peptide_data(config)
    
    if len(sequences) == 0:
        raise ValueError("No valid sequences found for calibration")
    
    # Use validation split for calibration
    _, val_loader, _ = create_data_loaders(sequences, labels, config, shuffle=False)
    
    return model, val_loader


def calibrate_temperature(model: nn.Module, val_loader: DataLoader, device: str, 
                         max_iter: int = 1000, lr: float = 0.01):
    """
    Calibrate temperature parameter on validation set.
    
    Args:
        model: Trained classifier model
        val_loader: Validation data loader
        device: Device to use
        max_iter: Maximum iterations for optimization
        lr: Learning rate
    
    Returns:
        Optimized temperature scaler
    """
    print("[calibrate_temperature] Starting temperature calibration...")
    
    # Collect logits and labels
    all_logits = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            sequences = batch['encoded'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(sequences)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    
    # Initialize temperature scaler
    temperature_scaler = TemperatureScaler().to(device)
    all_logits = all_logits.to(device)
    all_labels = all_labels.to(device)
    
    # Optimize temperature using negative log likelihood
    optimizer = torch.optim.LBFGS([temperature_scaler.temperature], lr=lr, max_iter=max_iter)
    
    def eval():
        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss()(temperature_scaler(all_logits), all_labels)
        loss.backward()
        return loss
    
    optimizer.step(eval)
    
    final_temperature = temperature_scaler.temperature.item()
    print(f"[calibrate_temperature] Calibrated temperature: {final_temperature:.4f}")
    
    return temperature_scaler


def evaluate_calibration(model: nn.Module, temperature_scaler: TemperatureScaler, 
                        val_loader: DataLoader, device: str):
    """Evaluate calibration before and after temperature scaling."""
    print("[calibrate_temperature] Evaluating calibration...")
    
    model.eval()
    temperature_scaler.eval()
    
    original_probs = []
    calibrated_probs = []
    labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            sequences = batch['encoded'].to(device)
            batch_labels = batch['label'].to(device)
            
            # Get original probabilities
            logits = model(sequences)
            orig_probs = torch.softmax(logits, dim=1)[:, 1]  # Positive class
            
            # Get calibrated probabilities
            cal_probs = temperature_scaler.calibrate_probs(logits)[:, 1]
            
            original_probs.extend(orig_probs.cpu().numpy())
            calibrated_probs.extend(cal_probs.cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())
    
    # Calculate expected calibration error (simplified)
    original_probs = np.array(original_probs)
    calibrated_probs = np.array(calibrated_probs)
    labels = np.array(labels)
    
    # Bin probabilities and calculate calibration error
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    
    orig_ece = _calculate_ece(original_probs, labels, bin_edges)
    cal_ece = _calculate_ece(calibrated_probs, labels, bin_edges)
    
    print(f"[calibrate_temperature] Expected Calibration Error:")
    print(f"  Original: {orig_ece:.4f}")
    print(f"  Calibrated: {cal_ece:.4f}")
    print(f"  Improvement: {orig_ece - cal_ece:.4f}")
    
    return {
        "original_ece": float(orig_ece),
        "calibrated_ece": float(cal_ece),
        "improvement": float(orig_ece - cal_ece)
    }


def _calculate_ece(probs: np.ndarray, labels: np.ndarray, bin_edges: np.ndarray) -> float:
    """Calculate expected calibration error."""
    ece = 0.0
    total_samples = len(probs)
    
    for i in range(len(bin_edges) - 1):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]
        
        # Find samples in this bin
        in_bin = (probs >= bin_lower) & (probs < bin_upper)
        if i == len(bin_edges) - 2:  # Include upper edge for last bin
            in_bin = (probs >= bin_lower) & (probs <= bin_upper)
        
        n_in_bin = np.sum(in_bin)
        
        if n_in_bin > 0:
            # Average predicted probability in bin
            avg_conf = np.mean(probs[in_bin])
            # Actual accuracy in bin
            avg_acc = np.mean(labels[in_bin])
            
            # Add to ECE
            ece += (n_in_bin / total_samples) * np.abs(avg_conf - avg_acc)
    
    return ece


def main():
    parser = argparse.ArgumentParser(description="Temperature calibration for classifier")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    
    # Load config
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"[calibrate_temperature] Error: Config file {args.config} not found")
        return 1
    
    # Setup device
    device = setup_device(config)
    
    try:
        # Load model and data
        model, val_loader = load_model_and_data(config, device)
        
        # Calibrate temperature
        temperature_scaler = calibrate_temperature(model, val_loader, device)
        
        # Evaluate calibration
        calibration_metrics = evaluate_calibration(model, temperature_scaler, val_loader, device)
        
        # Save temperature parameter
        temperature = temperature_scaler.temperature.item()
        temperature_data = {
            "temperature": temperature,
            "calibration_metrics": calibration_metrics
        }
        
        temperature_path = config["paths"]["temperature_json"]
        os.makedirs(os.path.dirname(temperature_path), exist_ok=True)
        
        with open(temperature_path, 'w') as f:
            json.dump(temperature_data, f, indent=2)
        
        print(f"[calibrate_temperature] Temperature parameter saved to {temperature_path}")
        
        # Save calibrated model (optional)
        calibrated_model_path = config["paths"]["classifier_ckpt"].replace(".pth", "_calibrated.pth")
        model.save(calibrated_model_path)
        print(f"[calibrate_temperature] Calibrated model saved to {calibrated_model_path}")
        
        return 0
        
    except Exception as e:
        print(f"[calibrate_temperature] Error during calibration: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
