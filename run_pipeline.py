#!/usr/bin/env python3
"""
One-click pipeline for BBB-penetrating peptide design.
Runs the complete pipeline: classifier training → temperature calibration → RL training → evaluation → saliency analysis.
"""

import os
import sys
import yaml
import argparse
import subprocess
import logging
from datetime import datetime
from pathlib import Path


def setup_logging(log_dir="results/logs"):
    """Setup logging for the pipeline."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"pipeline_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def check_required_files(config):
    """Check that required data files exist."""
    data_dir = config.get("paths", {}).get("data_dir", "data")
    required_files = [
        "Binding_peptide_sequence.csv",
        "negative_peptides_200.csv"
    ]
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        raise FileNotFoundError(f"Missing required data files: {missing_files}")


def create_default_config():
    """Create a default config.yaml file if it doesn't exist."""
    default_config = {
        "paths": {
            "data_dir": "data",
            "ckpt_dir": "checkpoints",
            "results_dir": "results",
            "runs_dir": "runs",
            "classifier_ckpt": "checkpoints/classifier_best.pth",
            "temperature_json": "results/temperature.json"
        },
        "classifier": {
            "seed": 42,
            "embedding": 128,
            "hidden": 256,
            "dropout": 0.3,
            "batch_size": 32,
            "epochs": 100,
            "lr": 0.001,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "min_cluster_identity": 0.8,
            "class_weights": [1.0, 1.0]
        },
        "rl": {
            "hidden_size": 256,
            "learning_rate": 0.0003,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_epsilon": 0.2,
            "entropy_coef": 0.01,
            "value_coef": 0.5,
            "max_grad_norm": 0.5,
            "ppo_epochs": 4,
            "batch_size": 64,
            "horizon": 2048
        },
        "task": {
            "min_len": 8,
            "max_len": 20
        },
        "physchem_constraints": {
            "max_cysteines": 2,
            "max_repeats": 3,
            "min_charge": -3,
            "max_charge": 6,
            "min_hydropathy": -2.0,
            "max_hydropathy": 2.0
        },
        "reward_weights": {
            "target_prob": 1.0,
            "specificity": 0.5,
            "physchem": 0.3,
            "motif": 0.2,
            "safety": 0.1
        },
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    with open("config.yaml", "w") as f:
        yaml.dump(default_config, f, default_flow_style=False, indent=2)
    
    return default_config


def ensure_directories(config):
    """Ensure all required directories exist."""
    paths = config.get("paths", {})
    directories = [
        paths.get("data_dir", "data"),
        paths.get("ckpt_dir", "checkpoints"),
        paths.get("results_dir", "results"),
        paths.get("runs_dir", "runs"),
        "results/logs",
        "results/saliency",
        "runs/ppo"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def run_command(cmd, logger):
    """Run a command and log the output."""
    logger.info(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Command failed: {cmd}")
            logger.error(f"Error output: {result.stderr}")
            return False
        logger.info(f"Command succeeded: {cmd}")
        if result.stdout:
            logger.info(f"Output: {result.stdout}")
        return True
    except Exception as e:
        logger.error(f"Exception running command {cmd}: {e}")
        return False


def find_latest_candidates(runs_dir="runs/ppo"):
    """Find the latest candidate file from RL training."""
    # Look for CSV files from RL training
    candidate_files = list(Path(runs_dir).glob("top_sequences_epoch_*.csv"))
    if not candidate_files:
        # Try any CSV files
        candidate_files = list(Path(runs_dir).glob("*.csv"))
    
    if not candidate_files:
        return None
    
    # Sort by modification time and get the latest
    latest_file = max(candidate_files, key=lambda x: x.stat().st_mtime)
    return str(latest_file)


def create_topk_candidates(input_file, output_file="runs/ppo/topk_candidates.csv", k=100):
    """Create topk candidates CSV from RL training CSV file."""
    if not os.path.exists(input_file):
        return False
    
    try:
        import pandas as pd
        
        # Read the CSV file
        df = pd.read_csv(input_file)
        
        # Check if 'sequence' column exists
        if 'sequence' not in df.columns:
            logging.error(f"CSV file {input_file} does not contain 'sequence' column")
            return False
        
        # Get top k sequences by reward (if reward column exists)
        if 'reward' in df.columns:
            top_candidates = df.nlargest(k, 'reward')
        else:
            top_candidates = df.head(k)
        
        # Save to output file
        top_candidates[['sequence']].to_csv(output_file, index=False)
        
        logging.info(f"Created top {len(top_candidates)} candidates from {input_file}")
        return True
    except Exception as e:
        logging.error(f"Error creating topk candidates: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run BBB-penetrating peptide design pipeline")
    parser.add_argument("--target", type=str, required=True, help="Target receptor name")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--limit", type=int, default=50, help="Limit for saliency analysis")
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    try:
        # Check if config exists, create default if not
        if not os.path.exists(args.config):
            logger.info("Config file not found, creating default config.yaml")
            config = create_default_config()
        else:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
        
        # Ensure directories exist
        ensure_directories(config)
        
        # Check required files
        check_required_files(config)
        
        logger.info("Starting BBB-penetrating peptide design pipeline")
        logger.info(f"Target receptor: {args.target}")
        logger.info(f"Using device: {config.get('device', 'cpu')}")
        
        # Step 1: Train classifier
        logger.info("Step 1: Training classifier")
        if not run_command(f"python -m src.train_classifier --config {args.config}", logger):
            logger.error("Classifier training failed")
            return 1
        
        # Step 2: Temperature calibration
        logger.info("Step 2: Temperature calibration")
        if not run_command(f"python -m src.calibrate_temperature --config {args.config}", logger):
            logger.error("Temperature calibration failed")
            return 1
        
        # Step 3: RL training
        logger.info("Step 3: RL training")
        if not run_command(f"python -m src.train_rl_ppo --config {args.config} --target {args.target}", logger):
            logger.error("RL training failed")
            return 1
        
        # Step 4: Prepare candidates for evaluation
        logger.info("Step 4: Preparing candidates for evaluation")
        latest_candidates = find_latest_candidates()
        if latest_candidates:
            if create_topk_candidates(latest_candidates):
                logger.info(f"Created topk candidates from {latest_candidates}")
            else:
                logger.error("Failed to create topk candidates")
                return 1
        else:
            logger.error("No candidate files found from RL training")
            return 1
        
        # Step 5: GPU evaluation
        logger.info("Step 5: GPU evaluation")
        if not run_command(
            f"python -m src.evaluate_candidates_gpu --config {args.config} --input runs/ppo/topk_candidates.csv --out results/eval_gpu.csv", 
            logger
        ):
            logger.error("GPU evaluation failed")
            return 1
        
        # Step 6: Positional saliency analysis
        logger.info("Step 6: Positional saliency analysis")
        if not run_command(
            f"python -m src.explain.positional_saliency --config {args.config} --input runs/ppo/topk_candidates.csv --target {args.target} --limit {args.limit} --outdir results/saliency", 
            logger
        ):
            logger.error("Positional saliency analysis failed")
            return 1
        
        logger.info("Pipeline completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        return 1


if __name__ == "__main__":
    # Import torch here to avoid issues in config creation
    try:
        import torch
    except ImportError:
        torch = None
    
    sys.exit(main())
