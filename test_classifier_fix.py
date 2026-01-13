"""
Test script to verify the classifier configuration fix.
"""

import sys
import yaml
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.classifier import PeptideClassifier

def test_classifier_loading():
    """Test that the classifier can load with the current config structure."""
    print("Testing classifier configuration compatibility...")
    
    try:
        # Load current config
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        print("Current config structure:")
        if "classifier" in config:
            print(f"  - classifier section exists")
            if "model" in config["classifier"]:
                print(f"  - classifier.model section exists")
                print(f"    embedding_dim: {config['classifier']['model'].get('embedding_dim', 'not set')}")
                print(f"    hidden_dim: {config['classifier']['model'].get('hidden_dim', 'not set')}")
                print(f"    dropout: {config['classifier']['model'].get('dropout', 'not set')}")
            else:
                print(f"  - classifier.model section NOT found")
        
        # Try to create classifier
        print("\nCreating classifier...")
        model = PeptideClassifier(config)
        print("[SUCCESS] Classifier created successfully!")
        
        # Test model parameters
        print(f"Model parameters:")
        print(f"  - embedding_dim: {model.embedding_dim}")
        print(f"  - hidden_dim: {model.hidden_dim}")
        print(f"  - dropout_rate: {model.dropout_rate}")
        print(f"  - max_length: {model.max_length}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_classifier_loading()
    sys.exit(0 if success else 1)
