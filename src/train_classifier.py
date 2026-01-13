#!/usr/bin/env python3
"""
Train supervised classifier for multi-receptor peptide binding prediction.
"""

import os
import sys
import yaml
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import json
from tqdm import tqdm

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.classifier_factory import build_classifier, log_model_config, count_parameters
from src.utils.data_loader import load_peptide_data, create_data_loaders, get_class_weights


def setup_device(config: dict) -> str:
    """Setup device for training."""
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("[train_classifier] CUDA not available, falling back to CPU")
        device = "cpu"
    
    print(f"[train_classifier] Using device: {device}")
    return device


def train_epoch(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, 
                optimizer: torch.optim.Optimizer, device: str) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    # Create progress bar for training
    pbar = tqdm(train_loader, desc="Training", leave=False)
    
    for batch in pbar:
        sequences = batch['encoded'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(sequences)
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return total_loss / len(train_loader)


def validate_epoch(model: nn.Module, val_loader: DataLoader, criterion: nn.Module, 
                  device: str) -> tuple:
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    # Create progress bar for validation
    pbar = tqdm(val_loader, desc="Validation", leave=False)
    
    with torch.no_grad():
        for batch in pbar:
            sequences = batch['encoded'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            logits = model(sequences)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            # Get predictions
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Positive class probabilities
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0
    
    avg_loss = total_loss / len(val_loader)
    
    return avg_loss, accuracy, f1, auc, all_preds, all_labels, all_probs


def print_metrics(epoch: int, train_loss: float, val_loss: float, 
                  accuracy: float, f1: float, auc: float, labels: list):
    """Print training metrics."""
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    
    print(f"Epoch {epoch:3d} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Acc: {accuracy:.4f} | "
          f"F1: {f1:.4f} | "
          f"AUC: {auc:.4f} | "
          f"Val samples: {len(labels)} ({n_pos}+/{n_neg}-)")


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                   epoch: int, loss: float, path: str):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


def main():
    parser = argparse.ArgumentParser(description="Train peptide classifier")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--target", type=str, default=None, help="Target receptor (optional)")
    args = parser.parse_args()
    
    # Load config
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"[train_classifier] Error: Config file {args.config} not found")
        return 1
    
    # Setup device
    device = setup_device(config)
    
    # Load data
    print("[train_classifier] Loading data...")
    sequences, labels = load_peptide_data(config, args.target)
    
    if len(sequences) == 0:
        print("[train_classifier] Error: No valid sequences found")
        return 1
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(sequences, labels, config)
    
    # Create model
    print("[train_classifier] Creating model...")
    log_model_config(config)
    model = build_classifier(config, device)
    n_params = count_parameters(model)
    print(f"[train_classifier] Model parameters: {n_params:,}")
    
    # Setup loss and optimizer
    classifier_config = config.get("classifier", {})
    class_weights = classifier_config.get("class_weights", [1.0, 1.0])
    if class_weights == "auto":
        class_weights = get_class_weights(labels).to(device)
    else:
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=classifier_config.get("lr", 0.001),
        weight_decay=1e-5
    )
    
    # Training loop
    print("[train_classifier] Starting training...")
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    epochs = classifier_config.get("epochs", 100)
    
    # Create progress bar for epochs
    epoch_pbar = tqdm(range(epochs), desc="Epochs", unit="epoch")
    
    for epoch in epoch_pbar:
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, accuracy, f1, auc, val_preds, val_labels, val_probs = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Update epoch progress bar
        epoch_pbar.set_postfix({
            "train_loss": f"{train_loss:.4f}",
            "val_loss": f"{val_loss:.4f}",
            "val_acc": f"{accuracy:.4f}",
            "val_f1": f"{f1:.4f}"
        })
        
        # Print detailed metrics
        print_metrics(epoch + 1, train_loss, val_loss, accuracy, f1, auc, val_labels)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save checkpoint
            ckpt_path = config["paths"]["classifier_ckpt"]
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            save_checkpoint(model, optimizer, epoch, val_loss, ckpt_path)
            print(f"[train_classifier] Saved best model to {ckpt_path}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"[train_classifier] Early stopping at epoch {epoch + 1}")
            break
    
    epoch_pbar.close()
    
    # Final evaluation on test set
    print("\n[train_classifier] Final evaluation on test set...")
    test_loss, test_accuracy, test_f1, test_auc, test_preds, test_labels, test_probs = validate_epoch(
        model, test_loader, criterion, device
    )
    
    # Calculate confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"Test Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_accuracy:.4f}")
    print(f"  F1 Score: {test_f1:.4f}")
    print(f"  AUC: {test_auc:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    True Negatives: {tn}")
    print(f"    False Positives: {fp}")
    print(f"    False Negatives: {fn}")
    print(f"    True Positives: {tp}")
    
    # Save final metrics
    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "test_f1": float(test_f1),
        "test_auc": float(test_auc),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        },
        "n_parameters": n_params
    }
    
    metrics_path = os.path.join(config["paths"]["results_dir"], "classifier_metrics.json")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"[train_classifier] Training completed. Metrics saved to {metrics_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
