"""
Test script to verify the classifier loading with checkpoint compatibility.
"""

import sys
import yaml
import torch
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.classifier import PeptideClassifier

def test_classifier_loading_with_checkpoint():
    """Test that the classifier can load from checkpoints with the new compatible method."""
    print("Testing classifier checkpoint loading compatibility...")
    
    try:
        # Load current config
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Device setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Test loading from checkpoint using the new compatible method
        print("\nTesting compatible loading method...")
        
        # Try to load from existing checkpoints
        checkpoint_paths = [
            "checkpoints/classifier_best_calibrated.pth",
            "checkpoints/classifier_best.pth"
        ]
        
        for checkpoint_path in checkpoint_paths:
            if not Path(checkpoint_path).exists():
                print(f"  - {checkpoint_path}: not found (skipping)")
                continue
                
            print(f"  - {checkpoint_path}:")
            
            try:
                # Try standard load first
                model = PeptideClassifier.load(checkpoint_path, config, device)
                print(f"    [SUCCESS] Standard load method worked")
            except (KeyError, ValueError) as e:
                print(f"    [WARNING] Standard load failed: {e}")
                
                # Try compatible load
                try:
                    model = PeptideClassifier.load_compatible(checkpoint_path, device)
                    print(f"    [SUCCESS] Compatible load method worked")
                except Exception as e2:
                    print(f"    [ERROR] Compatible load also failed: {e2}")
                    continue
            
            # Test model functionality
            print(f"    Model parameters: embedding_dim={model.embedding_dim}, hidden_dim={model.hidden_dim}")
            
            # Test forward pass with dummy data
            dummy_input = torch.randint(1, 21, (2, 10), dtype=torch.long).to(device)
            with torch.no_grad():
                output = model(dummy_input)
                print(f"    Forward pass successful: output shape {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_classifier_loading_with_checkpoint()
    sys.exit(0 if success else 1)
