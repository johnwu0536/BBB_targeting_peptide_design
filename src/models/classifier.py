"""
Peptide classifier model for multi-receptor binding prediction.
Uses embedding + LSTM architecture for sequence classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class PeptideClassifier(nn.Module):
    """LSTM-based classifier for peptide sequences."""
    
    def __init__(self, config: Dict = None, **kwargs):
        """
        Args:
            config: Configuration dictionary (optional)
            **kwargs: Additional parameters for backward compatibility
        """
        super().__init__()
        
        # Handle both config dict and direct parameters
        if config is not None:
            self.vocab_size = 21  # 20 AAs + padding
            
            # Handle different config structures for backward compatibility
            if "classifier" in config and "embedding" in config["classifier"]:
                # Old structure: config["classifier"]["embedding"]
                self.embedding_dim = config["classifier"]["embedding"]
                self.hidden_dim = config["classifier"]["hidden"]
                self.dropout_rate = config["classifier"]["dropout"]
                self.max_length = config["task"]["max_len"]
            elif "classifier" in config and "model" in config["classifier"]:
                # New structure: config["classifier"]["model"]["embedding_dim"]
                model_cfg = config["classifier"]["model"]
                self.embedding_dim = model_cfg.get("embedding_dim", 128)
                self.hidden_dim = model_cfg.get("hidden_dim", 256)
                self.dropout_rate = model_cfg.get("dropout", 0.3)
                self.max_length = config["task"]["max_len"]
            else:
                # Fallback to defaults
                self.embedding_dim = 128
                self.hidden_dim = 256
                self.dropout_rate = 0.3
                self.max_length = 20
        else:
            # Use kwargs with defaults
            self.vocab_size = kwargs.get("vocab_size", 21)
            self.embedding_dim = kwargs.get("embedding_dim", 128)
            self.hidden_dim = kwargs.get("hidden_dim", 256)
            self.dropout_rate = kwargs.get("dropout", 0.3)
            self.max_length = kwargs.get("max_length", 20)
        
        # Store extra config for compatibility
        self._extra_cfg = kwargs
        
        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=0  # padding token
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=self.dropout_rate,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(64, 2)  # binary classification
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len) with amino acid indices
        
        Returns:
            Logits of shape (batch_size, 2)
        """
        batch_size = x.size(0)
        
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)  # (batch_size, seq_len, hidden_dim * 2)
        
        # Attention mechanism
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)  # (batch_size, seq_len, 1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)  # (batch_size, hidden_dim * 2)
        
        # Classification
        logits = self.classifier(context_vector)  # (batch_size, 2)
        
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
        
        Returns:
            Probabilities of shape (batch_size, 2)
        """
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)
    
    def save(self, path: str):
        """Save model weights."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'embedding_dim': self.embedding_dim,
                'hidden_dim': self.hidden_dim,
                'dropout_rate': self.dropout_rate,
                'max_length': self.max_length
            }
        }, path)
    
    @classmethod
    def load(cls, path: str, config: Optional[Dict] = None, device: str = 'cpu'):
        """
        Load model from checkpoint.
        
        Args:
            path: Path to checkpoint
            config: Configuration dictionary (if None, will try to load from checkpoint)
            device: Device to load model on
        
        Returns:
            Loaded model
        """
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        
        if config is None:
            # Try to reconstruct config from checkpoint
            if 'config' in checkpoint:
                config = {
                    'classifier': {
                        'embedding': checkpoint['config']['embedding_dim'],
                        'hidden': checkpoint['config']['hidden_dim'],
                        'dropout': checkpoint['config']['dropout_rate']
                    },
                    'task': {
                        'max_len': checkpoint['config']['max_length']
                    }
                }
            else:
                raise ValueError("No config provided and no config found in checkpoint")
        
        # Create model with proper configuration handling
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        return model
    
    @classmethod
    def load_compatible(cls, path: str, device: str = 'cpu'):
        """
        Load model from checkpoint with maximum compatibility.
        This method handles cases where the config structure has changed.
        
        Args:
            path: Path to checkpoint
            device: Device to load model on
        
        Returns:
            Loaded model
        """
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        
        # Extract model parameters from checkpoint
        if 'config' in checkpoint:
            # Use parameters from checkpoint config
            checkpoint_config = checkpoint['config']
            embedding_dim = checkpoint_config.get('embedding_dim', 128)
            hidden_dim = checkpoint_config.get('hidden_dim', 256)
            dropout_rate = checkpoint_config.get('dropout_rate', 0.3)
            max_length = checkpoint_config.get('max_length', 20)
        else:
            # Use defaults
            embedding_dim = 128
            hidden_dim = 256
            dropout_rate = 0.3
            max_length = 20
        
        # Create model with direct parameters (bypassing config structure issues)
        model = cls(
            vocab_size=21,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout_rate,
            max_length=max_length
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        return model


def create_classifier(config: Dict, device: str = 'cpu') -> PeptideClassifier:
    """
    Create classifier model with proper device placement.
    
    Args:
        config: Configuration dictionary
        device: Device to place model on
    
    Returns:
        Classifier model
    """
    model = PeptideClassifier(config)
    model.to(device)
    return model


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
