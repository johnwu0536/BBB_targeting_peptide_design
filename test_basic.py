#!/usr/bin/env python3
"""
Basic test script to verify the pipeline structure without requiring all dependencies.
"""

import os
import sys
import yaml


def test_config():
    """Test that config.yaml is valid."""
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


def test_file_structure():
    """Test that all required files exist."""
    print("\nTesting file structure...")
    
    required_files = [
        "run_pipeline.py",
        "config.yaml",
        "test_pipeline.py",
        "requirements.txt",
        "README.md",
        "src/__init__.py",
        "src/utils/__init__.py",
        "src/utils/encoding.py",
        "src/utils/data_loader.py",
        "src/models/__init__.py",
        "src/models/classifier.py",
        "src/train_classifier.py",
        "src/calibrate_temperature.py",
        "src/explain/__init__.py",
        "src/explain/positional_saliency.py",
        "baseline_L1_classic_opt/__init__.py",
        "baseline_L1_classic_opt/src/__init__.py",
        "baseline_L1_classic_opt/src/aa.py",
        "baseline_L1_classic_opt/src/score.py",
        "baseline_L1_classic_opt/src/ga_optimize.py",
        "baseline_L1_classic_opt/src/run_l1.py"
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"[OK] Found: {file_path}")
        else:
            print(f"[ERROR] Missing: {file_path}")
            all_exist = False
    
    return all_exist


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
        else:
            print(f"[ERROR] Missing: {file_path}")
            all_exist = False
    
    return all_exist


def test_syntax():
    """Test basic Python syntax of key files."""
    print("\nTesting Python syntax...")
    
    files_to_test = [
        "src/utils/encoding.py",
        "baseline_L1_classic_opt/src/aa.py",
        "baseline_L1_classic_opt/src/score.py"
    ]
    
    all_valid = True
    for file_path in files_to_test:
        try:
            with open(file_path, 'r') as f:
                source = f.read()
            compile(source, file_path, 'exec')
            print(f"[OK] Valid syntax: {file_path}")
        except SyntaxError as e:
            print(f"[ERROR] Syntax error in {file_path}: {e}")
            all_valid = False
    
    return all_valid


def main():
    """Run all basic tests."""
    print("=" * 60)
    print("BBB-Penetrating Peptide Design Pipeline - Basic Test")
    print("=" * 60)
    
    tests = [
        test_config,
        test_file_structure,
        test_data_files,
        test_syntax
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
    print("BASIC TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("[SUCCESS] Basic structure tests passed!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run full test: python test_pipeline.py")
        print("3. Run pipeline: python run_pipeline.py --target <RECEPTOR> --config config.yaml")
    else:
        print("[WARNING] Some basic tests failed. Please check the errors above.")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
