#!/usr/bin/env python3
"""
Run L1 baseline: Classic sequence optimization without AI.
"""

import os
import sys
import yaml
import argparse
from typing import List, Tuple

# Add baseline to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .ga_optimize import run_genetic_algorithm, save_ga_results
from .score import get_detailed_scores


def main():
    parser = argparse.ArgumentParser(description="Run L1 baseline optimization")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--output", type=str, default="results/l1_baseline.csv", help="Output file")
    parser.add_argument("--generations", type=int, default=100, help="Number of generations")
    parser.add_argument("--population", type=int, default=100, help="Population size")
    args = parser.parse_args()
    
    # Load config
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"[L1] Error: Config file {args.config} not found")
        return 1
    
    # Extract constraints and weights
    constraints = config.get("physchem_constraints", {})
    reward_weights = config.get("reward_weights", {})
    
    # Add length constraints from task config
    task_config = config.get("task", {})
    constraints["min_len"] = task_config.get("min_len", 8)
    constraints["max_len"] = task_config.get("max_len", 20)
    
    print("[L1] Starting L1 baseline optimization")
    print(f"[L1] Constraints: {constraints}")
    print(f"[L1] Reward weights: {reward_weights}")
    
    # Run genetic algorithm
    results = run_genetic_algorithm(
        constraints=constraints,
        reward_weights=reward_weights,
        generations=args.generations,
        population_size=args.population
    )
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_ga_results(results, args.output, constraints)
    
    # Print top results with detailed scores
    print("\n[L1] Top 5 sequences:")
    for i, (seq, score) in enumerate(results[:5]):
        detailed = get_detailed_scores(seq, constraints, reward_weights)
        print(f"  {i+1}. {seq} (score: {score:.4f})")
        print(f"     Components: {detailed['components']}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
