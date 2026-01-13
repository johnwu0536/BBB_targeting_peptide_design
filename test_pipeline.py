f#!/usr/bin/env python3
"""
Test script to verify the BBB-penetrating peptide design pipeline.
This script tests basic functionality without running full training.
"""

import os
import sys
import yaml
import torch
import pandas as pd
from pathlib import Path


def test_config():
    """Test that config.yaml is valid and contains required keys."""
    print("Testing config.yaml...")
    
    try:
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        required_sections = ["paths", "classifier", "rl", "task", "physchem_constraints", "reward_weights"]
        for section in required_sections:
            if section not in config:
                print(f"[ERROR] Missing required section: {section}")
                return False
            else:
                print(f"[OK] Found section: {section}")
        
        print("[OK] Config file is valid")
        return True
        
    except Exception as e:
        print(f"[ERROR] Error loading config: {e}")
        return False


def test_data_files():
    """Test that required data files exist."""
    print("\nTesting data files...")
    
    required_files = [
        "data/Binding_peptide_sequence.csv",
        "data/negative_peptides_200.csv",
        "data/ecto_domain_BBB_receptor.csv"
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"[OK] Found: {file_path}")
            
            # Check file content
            try:
                df = pd.read_csv(file_path)
                print(f"   - Rows: {len(df)}, Columns: {len(df.columns)}")
            except Exception as e:
                print(f"   [WARNING] Could not read: {e}")
        else:
            print(f"[ERROR] Missing: {file_path}")
            all_exist = False
    
    return all_exist


def test_imports():
    """Test that all modules can be imported."""
    print("\nTesting module imports...")
    
    modules_to_test = [
        "src.utils.encoding",
        "src.utils.data_loader", 
        "src.models.classifier",
        "src.train_classifier",
        "src.calibrate_temperature",
        "src.explain.positional_saliency",
        "baseline_L1_classic_opt.src.aa",
        "baseline_L1_classic_opt.src.score",
        "baseline_L1_classic_opt.src.ga_optimize",
        "baseline_L1_classic_opt.src.run_l1"
    ]
    
    all_imported = True
    for module_path in modules_to_test:
        try:
            __import__(module_path)
            print(f"[OK] Imported: {module_path}")
        except ImportError as e:
            print(f"[ERROR] Failed to import {module_path}: {e}")
            all_imported = False
    
    return all_imported


def test_encoding():
    """Test sequence encoding functionality."""
    print("\nTesting sequence encoding...")
    
    try:
        from src.utils.encoding import clean_sequence, encode_sequence, AA_ALPHABET
        
        # Test sequence cleaning
        test_sequences = [
            "ACDEFGH",  # Valid
            "ACDEFGHX",  # Contains invalid character
            "acdefgh",   # Lowercase
            "ACD EFG H"  # Contains spaces
        ]
        
        for seq in test_sequences:
            cleaned = clean_sequence(seq, strict=False)
            print(f"   '{seq}' -> '{cleaned}'")
        
        # Test encoding
        test_seq = "ACDEFG"
        encoded = encode_sequence(test_seq, max_length=10)
        print(f"   Encoded '{test_seq}': shape {encoded.shape}")
        
        print("[OK] Encoding tests passed")
        return True
        
    except Exception as e:
        print(f"[ERROR] Encoding test failed: {e}")
        return False


def test_l1_baseline():
    """Test L1 baseline functionality."""
    print("\nTesting L1 baseline...")
    
    try:
        from baseline_L1_classic_opt.src.aa import (
            calculate_hydropathy, calculate_charge, count_cysteines,
            generate_random_sequence, validate_sequence
        )
        from baseline_L1_classic_opt.src.score import score_sequence
        
        # Test sequence generation
        test_seq = generate_random_sequence(10)
        print(f"   Generated sequence: {test_seq}")
        
        # Test validation
        is_valid = validate_sequence(test_seq)
        print(f"   Sequence valid: {is_valid}")
        
        # Test scoring
        constraints = {
            "min_hydropathy": -2.0,
            "max_hydropathy": 2.0,
            "min_charge": -3,
            "max_charge": 6,
            "max_cysteines": 2,
            "max_repeats": 3
        }
        reward_weights = {"physchem": 1.0, "motif": 0.5}
        
        score = score_sequence(test_seq, constraints, reward_weights)
        print(f"   Sequence score: {score:.4f}")
        
        print("[OK] L1 baseline tests passed")
        return True
        
    except Exception as e:
        print(f"[ERROR] L1 baseline test failed: {e}")
        return False


def test_device_setup():
    """Test device setup for GPU/CPU."""
    print("\nTesting device setup...")
    
    try:
        # Test CUDA availability
        cuda_available = torch.cuda.is_available()
        print(f"   CUDA available: {cuda_available}")
        
        if cuda_available:
            print(f"   CUDA device count: {torch.cuda.device_count()}")
            print(f"   Current device: {torch.cuda.current_device()}")
            print(f"   Device name: {torch.cuda.get_device_name()}")
        
        # Test basic tensor operations
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = x + y
        print(f"   Tensor operations work: {z.shape}")
        
        print("[OK] Device setup tests passed")
        return True
        
    except Exception as e:
        print(f"[ERROR] Device setup test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("BBB-Penetrating Peptide Design Pipeline Test")
    print("=" * 60)
    
    tests = [
        test_config,
        test_data_files,
        test_imports,
        test_encoding,
        test_l1_baseline,
        test_device_setup
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"[ERROR] Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("[SUCCESS] All tests passed! The pipeline is ready to use.")
        print("\nNext steps:")
        print("1. Run the full pipeline: python run_pipeline.py --target <RECEPTOR>")
        print("2. Test L1 baseline: python -m baseline_L1_classic_opt.src.run_l1 --config config.yaml")
        print("3. Train classifier: python -m src.train_classifier --config config.yaml")
    else:
        print("[WARNING] Some tests failed. Please check the errors above.")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
