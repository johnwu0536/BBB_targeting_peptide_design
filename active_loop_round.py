#!/usr/bin/env python3
"""
Active Learning Orchestration Script for BBB-Penetrating Peptide Design.

Implements one full active learning iteration:
1. Retrain classifier if new labeled data exists
2. Recalibrate classifier with temperature scaling
3. Run PPO RL to generate new peptide candidates
4. Evaluate candidates with calibrated classifier
5. Use Bayesian Optimization to select top candidates
6. Log results and prepare for next round
"""

import os
import sys
import argparse
import yaml
import pandas as pd
import torch
from pathlib import Path
import logging
import json
from datetime import datetime
from typing import Dict

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from src.train_classifier import main as train_classifier
from src.calibrate_temperature import main as calibrate_temperature
from src.train_rl_ppo import main as train_rl_ppo
from src.evaluate_candidates_gpu import main as evaluate_candidates
from src.active_selection.bo_selector import BOSelector

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(name)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"results/logs/active_learning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ActiveLearningOrchestrator:
    """Orchestrates the active learning pipeline for peptide design."""
    
    def __init__(self, config: dict, target_receptor: str, round_number: int = 1):
        self.config = config
        self.target_receptor = target_receptor
        self.round_number = round_number
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup directories
        self._setup_directories()
        
        logger.info(f"Active Learning Round {round_number} initialized for target: {target_receptor}")
    
    def _setup_directories(self):
        """Create necessary directories for active learning."""
        directories = [
            "results/active_learning",
            "runs/active_learning",
            "checkpoints/active_learning"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def check_new_data(self) -> bool:
        """
        Check if new labeled data is available for retraining.
        
        Returns:
            True if new data is available, False otherwise
        """
        # This is a placeholder - in practice, you would check for new experimental results
        # For now, we'll assume no new data in the first few rounds
        if self.round_number <= 1:
            logger.info("No new experimental data available (first round)")
            return False
        
        # Check for new data file
        new_data_file = f"data/new_experimental_round_{self.round_number - 1}.csv"
        if os.path.exists(new_data_file):
            logger.info(f"New experimental data found: {new_data_file}")
            return True
        
        logger.info("No new experimental data found")
        return False
    
    def retrain_classifier(self) -> bool:
        """
        Retrain classifier with new data if available.
        
        Returns:
            True if retraining was successful, False otherwise
        """
        if not self.check_new_data():
            logger.info("Skipping classifier retraining - no new data")
            return True
        
        try:
            logger.info("Starting classifier retraining with new data...")
            
            # Update config to include new data
            updated_config = self.config.copy()
            # Here you would modify the config to include new data paths
            
            # Run classifier training
            train_classifier(updated_config)
            
            logger.info("Classifier retraining completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Classifier retraining failed: {e}")
            return False
    
    def _check_classifier_checkpoints(self) -> Dict[str, bool]:
        """
        Check existence of base and calibrated classifier checkpoints.
        
        Returns:
            Dictionary with checkpoint existence status
        """
        base_checkpoint = self.config["paths"]["classifier_ckpt"]
        calibrated_checkpoint = self.config.get("classifier", {}).get("calibrated_checkpoint", 
                                                                     "checkpoints/classifier_best_calibrated.pth")
        
        status = {
            "base_exists": os.path.exists(base_checkpoint),
            "calibrated_exists": os.path.exists(calibrated_checkpoint),
            "base_path": base_checkpoint,
            "calibrated_path": calibrated_checkpoint
        }
        
        logger.info(f"Classifier checkpoints - Base: {status['base_exists']}, Calibrated: {status['calibrated_exists']}")
        return status
    
    def _run_calibration_safe(self) -> bool:
        """
        Run classifier calibration with safe exception handling.
        
        Returns:
            True if calibration was successful, False otherwise
        """
        try:
            logger.info("Starting classifier calibration...")
            
            # Run temperature calibration via command line
            import subprocess
            result = subprocess.run([
                "python", "-m", "src.calibrate_temperature", 
                "--config", "config.yaml"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Classifier calibration completed successfully")
                return True
            else:
                logger.error(f"Classifier calibration failed: {result.stderr}")
                return False
            
        except Exception as e:
            import traceback
            logger.error(f"Classifier calibration failed with exception: {repr(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def recalibrate_classifier(self) -> bool:
        """
        Recalibrate classifier with robust checkpoint handling.
        
        Returns:
            True if calibration was successful or can proceed without it
        """
        checkpoint_status = self._check_classifier_checkpoints()
        
        # If no base checkpoint exists, we cannot proceed
        if not checkpoint_status["base_exists"]:
            logger.error(f"No base classifier checkpoint found at {checkpoint_status['base_path']}")
            logger.error("Cannot proceed without a trained classifier. Please run classifier training first.")
            return False
        
        # If calibrated checkpoint already exists and no new data, reuse it
        if checkpoint_status["calibrated_exists"] and not self.check_new_data():
            logger.info("Calibrated classifier already exists and no new data available - reusing existing calibrated model")
            return True
        
        # If we need calibration, run it safely
        logger.info("Attempting classifier calibration...")
        calibration_success = self._run_calibration_safe()
        
        if calibration_success:
            logger.info("Classifier calibration completed successfully")
            return True
        else:
            # If calibration fails but we have a base model, we can proceed with warning
            if checkpoint_status["base_exists"]:
                logger.warning("Classifier calibration failed, but base model exists - proceeding with uncalibrated classifier")
                return True
            else:
                logger.error("Classifier calibration failed and no base model available - cannot proceed")
                return False
    
    def run_rl_generation(self) -> bool:
        """
        Run RL to generate new peptide candidates.
        
        Returns:
            True if RL generation was successful, False otherwise
        """
        try:
            logger.info("Starting RL candidate generation...")
            
            # Run RL training via command line
            import subprocess
            result = subprocess.run([
                "python", "-m", "src.train_rl_ppo", 
                "--config", "config.yaml",
                "--target", self.target_receptor,
                "--epochs", "500",
                "--batch_size", "32"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                # Consolidate candidate files
                self._consolidate_candidates()
                logger.info("RL candidate generation completed successfully")
                return True
            else:
                logger.error(f"RL candidate generation failed: {result.stderr}")
                return False
            
        except Exception as e:
            logger.error(f"RL candidate generation failed: {e}")
            return False
    
    def _consolidate_candidates(self):
        """Consolidate candidate files from RL training."""
        import glob
        
        # Find all candidate files
        candidate_files = glob.glob("runs/ppo/top_sequences_epoch_*.csv")
        candidate_files.append("runs/ppo/top_sequences_epoch_final.csv")
        
        all_candidates = []
        
        for file in candidate_files:
            if os.path.exists(file):
                try:
                    df = pd.read_csv(file)
                    all_candidates.append(df)
                except Exception as e:
                    logger.warning(f"Could not read candidate file {file}: {e}")
        
        if all_candidates:
            # Combine and deduplicate
            combined_df = pd.concat(all_candidates, ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['sequence'])
            
            # Save consolidated candidates
            output_file = f"runs/active_learning/candidates_round_{self.round_number}_raw.csv"
            combined_df.to_csv(output_file, index=False)
            logger.info(f"Consolidated {len(combined_df)} unique candidates to {output_file}")
        else:
            logger.warning("No candidate files found to consolidate")
    
    def evaluate_candidates(self) -> bool:
        """
        Evaluate generated candidates with calibrated classifier.
        
        Returns:
            True if evaluation was successful, False otherwise
        """
        try:
            logger.info("Starting candidate evaluation...")
            
            input_file = f"runs/active_learning/candidates_round_{self.round_number}_raw.csv"
            output_file = f"results/active_learning/evaluated_round_{self.round_number}.csv"
            
            if not os.path.exists(input_file):
                logger.error(f"Candidate file not found: {input_file}")
                return False
            
            # Run evaluation
            evaluate_candidates(
                config=self.config,
                input_file=input_file,
                output_file=output_file
            )
            
            logger.info(f"Candidate evaluation completed, results saved to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Candidate evaluation failed: {e}")
            return False
    
    def select_candidates_with_bo(self) -> bool:
        """
        Select top candidates using Bayesian Optimization.
        
        Returns:
            True if selection was successful, False otherwise
        """
        try:
            logger.info("Starting Bayesian Optimization candidate selection...")
            
            input_file = f"results/active_learning/evaluated_round_{self.round_number}.csv"
            output_file = f"results/active_learning/selected_round_{self.round_number}.csv"
            
            if not os.path.exists(input_file):
                logger.error(f"Evaluated candidate file not found: {input_file}")
                return False
            
            # Run BO selection
            selector = BOSelector(self.config)
            selected_candidates = selector.select_candidates(input_file, output_file)
            
            if selected_candidates.empty:
                logger.error("No candidates selected by BO")
                return False
            
            logger.info(f"BO selection completed, {len(selected_candidates)} candidates selected")
            return True
            
        except Exception as e:
            logger.error(f"BO candidate selection failed: {e}")
            return False
    
    def log_round_statistics(self):
        """Log statistics for the current active learning round."""
        try:
            selected_file = f"results/active_learning/selected_round_{self.round_number}.csv"
            
            if not os.path.exists(selected_file):
                logger.warning("No selected candidates file found for statistics")
                return
            
            df = pd.read_csv(selected_file)
            
            stats = {
                "round": self.round_number,
                "target": self.target_receptor,
                "timestamp": datetime.now().isoformat(),
                "n_candidates": len(df),
                "avg_acquisition_score": df['acquisition_score'].mean() if 'acquisition_score' in df.columns else 0,
                "avg_uncertainty": df['uncertainty'].mean() if 'uncertainty' in df.columns else 0,
                "high_confidence_count": len(df[df.get('acquisition_score', 0) >= 0.8]) if 'acquisition_score' in df.columns else 0
            }
            
            # Save statistics
            stats_file = f"results/active_learning/round_{self.round_number}_stats.json"
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            
            logger.info(f"Round {self.round_number} statistics: {stats}")
            
        except Exception as e:
            logger.error(f"Error logging round statistics: {e}")
    
    def run_round(self) -> bool:
        """
        Run one full active learning round.
        
        Returns:
            True if round completed successfully, False otherwise
        """
        logger.info(f"=== Starting Active Learning Round {self.round_number} ===")
        
        # Step 1: Retrain classifier if new data available
        if not self.retrain_classifier():
            logger.warning("Classifier retraining failed, continuing with existing model")
        
        # Step 2: Recalibrate classifier
        if not self.recalibrate_classifier():
            logger.error("Classifier calibration failed")
            return False
        
        # Step 3: Generate candidates with RL
        if not self.run_rl_generation():
            logger.error("RL candidate generation failed")
            return False
        
        # Step 4: Evaluate candidates
        if not self.evaluate_candidates():
            logger.error("Candidate evaluation failed")
            return False
        
        # Step 5: Select candidates with BO
        if not self.select_candidates_with_bo():
            logger.error("BO candidate selection failed")
            return False
        
        # Step 6: Log statistics
        self.log_round_statistics()
        
        logger.info(f"=== Active Learning Round {self.round_number} Completed Successfully ===")
        return True


def main():
    parser = argparse.ArgumentParser(description="Active Learning Orchestration for Peptide Design")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--target", type=str, required=True, help="Target receptor")
    parser.add_argument("--round", type=int, default=1, help="Active learning round number")
    
    args = parser.parse_args()
    
    # Load config
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading config file {args.config}: {e}")
        return 1
    
    # Create orchestrator and run round
    orchestrator = ActiveLearningOrchestrator(config, args.target, args.round)
    
    if orchestrator.run_round():
        logger.info("Active learning round completed successfully")
        return 0
    else:
        logger.error("Active learning round failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
