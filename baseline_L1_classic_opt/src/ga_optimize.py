"""
Genetic algorithm optimization for peptide sequences.
"""

import random
import numpy as np
from typing import List, Tuple, Dict
from tqdm import tqdm

from .aa import (
    generate_random_sequence, mutate_sequence, crossover_sequences,
    validate_sequence, get_sequence_stats
)
from .score import score_sequence, validate_sequence_constraints


class GeneticAlgorithm:
    """Genetic algorithm for peptide sequence optimization."""
    
    def __init__(self, constraints: dict, reward_weights: dict):
        """
        Args:
            constraints: Physicochemical constraints
            reward_weights: Weights for scoring components
        """
        self.constraints = constraints
        self.reward_weights = reward_weights
        self.min_len = constraints.get("min_len", 8)
        self.max_len = constraints.get("max_len", 20)
        
        # GA parameters
        self.population_size = 100
        self.generations = 100
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.elitism_count = 5
    
    def initialize_population(self) -> List[str]:
        """Initialize random population."""
        population = []
        while len(population) < self.population_size:
            length = random.randint(self.min_len, self.max_len)
            seq = generate_random_sequence(length)
            if validate_sequence_constraints(seq, self.constraints):
                population.append(seq)
        return population
    
    def evaluate_population(self, population: List[str]) -> List[Tuple[str, float]]:
        """Evaluate and score population."""
        scored_population = []
        for seq in population:
            score = score_sequence(seq, self.constraints, self.reward_weights)
            scored_population.append((seq, score))
        
        # Sort by score (descending)
        scored_population.sort(key=lambda x: x[1], reverse=True)
        return scored_population
    
    def select_parents(self, scored_population: List[Tuple[str, float]]) -> List[str]:
        """Select parents using tournament selection."""
        parents = []
        
        # Always keep the best individuals (elitism)
        elite = [seq for seq, score in scored_population[:self.elitism_count]]
        parents.extend(elite)
        
        # Tournament selection for the rest
        tournament_size = 3
        while len(parents) < self.population_size:
            tournament = random.sample(scored_population, tournament_size)
            winner = max(tournament, key=lambda x: x[1])
            parents.append(winner[0])
        
        return parents
    
    def crossover_population(self, parents: List[str]) -> List[str]:
        """Perform crossover to create new population."""
        new_population = []
        
        # Keep elites
        new_population.extend(parents[:self.elitism_count])
        
        # Create offspring through crossover
        while len(new_population) < self.population_size:
            parent1, parent2 = random.sample(parents, 2)
            
            if random.random() < self.crossover_rate:
                child = crossover_sequences(parent1, parent2)
            else:
                # No crossover, randomly select one parent
                child = random.choice([parent1, parent2])
            
            # Validate child
            if validate_sequence_constraints(child, self.constraints):
                new_population.append(child)
            else:
                # If invalid, add a random valid sequence
                length = random.randint(self.min_len, self.max_len)
                new_seq = generate_random_sequence(length)
                if validate_sequence_constraints(new_seq, self.constraints):
                    new_population.append(new_seq)
        
        return new_population
    
    def mutate_population(self, population: List[str]) -> List[str]:
        """Apply mutation to population."""
        mutated_population = []
        
        for seq in population:
            if random.random() < self.mutation_rate:
                mutated = mutate_sequence(seq, self.mutation_rate)
                # Validate mutated sequence
                if validate_sequence_constraints(mutated, self.constraints):
                    mutated_population.append(mutated)
                else:
                    mutated_population.append(seq)  # Keep original if invalid
            else:
                mutated_population.append(seq)
        
        return mutated_population
    
    def run(self, verbose: bool = True) -> Tuple[List[Tuple[str, float]], Dict]:
        """Run genetic algorithm optimization."""
        # Initialize population
        population = self.initialize_population()
        best_scores = []
        diversity_scores = []
        
        if verbose:
            print(f"[GA] Starting optimization with population size {self.population_size}")
            print(f"[GA] Running for {self.generations} generations")
        
        for generation in tqdm(range(self.generations), disable=not verbose):
            # Evaluate population
            scored_population = self.evaluate_population(population)
            
            # Track best score
            best_seq, best_score = scored_population[0]
            best_scores.append(best_score)
            
            # Track diversity
            diversity = len(set(population)) / len(population)
            diversity_scores.append(diversity)
            
            if verbose and generation % 10 == 0:
                print(f"[GA] Generation {generation}: Best score = {best_score:.4f}, "
                      f"Diversity = {diversity:.3f}")
                print(f"[GA] Best sequence: {best_seq}")
            
            # Check for convergence
            if generation > 20 and len(set(best_scores[-10:])) == 1:
                if verbose:
                    print(f"[GA] Convergence detected at generation {generation}")
                break
            
            # Selection
            parents = self.select_parents(scored_population)
            
            # Crossover
            offspring = self.crossover_population(parents)
            
            # Mutation
            population = self.mutate_population(offspring)
        
        # Final evaluation
        final_scored = self.evaluate_population(population)
        
        # Statistics
        stats = {
            "best_scores": best_scores,
            "diversity_scores": diversity_scores,
            "final_population_size": len(population),
            "unique_sequences": len(set(population))
        }
        
        return final_scored, stats


def run_genetic_algorithm(constraints: dict, reward_weights: dict, 
                         generations: int = 100, population_size: int = 100) -> List[Tuple[str, float]]:
    """
    Run genetic algorithm optimization.
    
    Args:
        constraints: Physicochemical constraints
        reward_weights: Weights for scoring components
        generations: Number of generations
        population_size: Population size
    
    Returns:
        List of (sequence, score) tuples sorted by score
    """
    ga = GeneticAlgorithm(constraints, reward_weights)
    ga.generations = generations
    ga.population_size = population_size
    
    results, stats = ga.run(verbose=True)
    
    print(f"\n[GA] Optimization completed:")
    print(f"  Best score: {results[0][1]:.4f}")
    print(f"  Best sequence: {results[0][0]}")
    print(f"  Final diversity: {stats['diversity_scores'][-1]:.3f}")
    print(f"  Unique sequences: {stats['unique_sequences']}")
    
    return results


def save_ga_results(results: List[Tuple[str, float]], output_file: str, constraints: dict):
    """Save GA results to file."""
    import csv
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sequence', 'score', 'length', 'hydropathy', 'charge', 'cysteines', 'repeats'])
        
        for seq, score in results:
            stats = get_sequence_stats(seq)
            writer.writerow([
                seq, score, stats['length'], stats['hydropathy'], 
                stats['charge'], stats['cysteines'], stats['repeats']
            ])
    
    print(f"[GA] Results saved to {output_file}")
