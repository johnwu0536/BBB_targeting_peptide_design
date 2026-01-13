"""
Scoring function for L1 baseline optimization.
Hand-crafted scoring using KD hydropathy, net charge, repeat penalty, etc.
"""

from .aa import (
    calculate_hydropathy, calculate_charge, count_cysteines, 
    count_repeats, contains_motif
)


def score_sequence(sequence: str, constraints: dict, reward_weights: dict) -> float:
    """
    Score a peptide sequence using hand-crafted scoring function.
    
    Args:
        sequence: Peptide sequence to score
        constraints: Physicochemical constraints
        reward_weights: Weights for different scoring components
    
    Returns:
        Total score (higher is better)
    """
    if not sequence:
        return 0.0
    
    # Calculate individual components
    hydropathy_score = _score_hydropathy(sequence, constraints)
    charge_score = _score_charge(sequence, constraints)
    cysteine_penalty = _score_cysteines(sequence, constraints)
    repeat_penalty = _score_repeats(sequence, constraints)
    motif_bonus = _score_motifs(sequence)
    
    # Combine scores with weights
    total_score = (
        reward_weights.get("physchem", 1.0) * (
            hydropathy_score + charge_score - cysteine_penalty - repeat_penalty
        ) +
        reward_weights.get("motif", 0.5) * motif_bonus
    )
    
    return total_score


def _score_hydropathy(sequence: str, constraints: dict) -> float:
    """Score sequence based on hydropathy constraints."""
    hydropathy = calculate_hydropathy(sequence)
    min_hydropathy = constraints.get("min_hydropathy", -2.0)
    max_hydropathy = constraints.get("max_hydropathy", 2.0)
    
    # Penalize if outside desired range
    if hydropathy < min_hydropathy:
        return -abs(hydropathy - min_hydropathy)
    elif hydropathy > max_hydropathy:
        return -abs(hydropathy - max_hydropathy)
    else:
        # Reward being in the middle of the range
        target = (min_hydropathy + max_hydropathy) / 2
        return 1.0 - abs(hydropathy - target) / (max_hydropathy - min_hydropathy)


def _score_charge(sequence: str, constraints: dict) -> float:
    """Score sequence based on charge constraints."""
    charge = calculate_charge(sequence)
    min_charge = constraints.get("min_charge", -3)
    max_charge = constraints.get("max_charge", 6)
    
    # Penalize if outside desired range
    if charge < min_charge:
        return -abs(charge - min_charge)
    elif charge > max_charge:
        return -abs(charge - max_charge)
    else:
        # Reward moderate positive charge (often beneficial for BBB penetration)
        target_charge = 2  # Slightly positive
        return 1.0 - abs(charge - target_charge) / (max_charge - min_charge)


def _score_cysteines(sequence: str, constraints: dict) -> float:
    """Penalize excessive cysteine residues."""
    cysteines = count_cysteines(sequence)
    max_cysteines = constraints.get("max_cysteines", 2)
    
    if cysteines <= max_cysteines:
        return 0.0
    else:
        # Quadratic penalty for excessive cysteines
        excess = cysteines - max_cysteines
        return excess ** 2


def _score_repeats(sequence: str, constraints: dict) -> float:
    """Penalize repeating amino acid patterns."""
    repeats = count_repeats(sequence)
    max_repeats = constraints.get("max_repeats", 3)
    
    if repeats <= max_repeats:
        return 0.0
    else:
        # Linear penalty for excessive repeats
        excess = repeats - max_repeats
        return excess * 0.5


def _score_motifs(sequence: str) -> float:
    """Bonus for containing known BBB penetration motifs."""
    motif_count = contains_motif(sequence)
    return motif_count * 0.5  # Small bonus per motif


def validate_sequence_constraints(sequence: str, constraints: dict) -> bool:
    """
    Validate if sequence meets all hard constraints.
    
    Args:
        sequence: Peptide sequence to validate
        constraints: Physicochemical constraints
    
    Returns:
        True if sequence meets all constraints
    """
    # Check length constraints
    min_len = constraints.get("min_len", 8)
    max_len = constraints.get("max_len", 20)
    
    if len(sequence) < min_len or len(sequence) > max_len:
        return False
    
    # Check cysteine constraint
    cysteines = count_cysteines(sequence)
    max_cysteines = constraints.get("max_cysteines", 2)
    if cysteines > max_cysteines:
        return False
    
    # Check hydropathy constraint (soft constraint, but can be made hard)
    hydropathy = calculate_hydropathy(sequence)
    min_hydropathy = constraints.get("min_hydropathy", -2.0)
    max_hydropathy = constraints.get("max_hydropathy", 2.0)
    
    if hydropathy < min_hydropathy or hydropathy > max_hydropathy:
        return False
    
    # Check charge constraint
    charge = calculate_charge(sequence)
    min_charge = constraints.get("min_charge", -3)
    max_charge = constraints.get("max_charge", 6)
    
    if charge < min_charge or charge > max_charge:
        return False
    
    return True


def get_detailed_scores(sequence: str, constraints: dict, reward_weights: dict) -> dict:
    """
    Get detailed breakdown of scoring components.
    
    Args:
        sequence: Peptide sequence to score
        constraints: Physicochemical constraints
        reward_weights: Weights for different scoring components
    
    Returns:
        Dictionary with detailed scoring breakdown
    """
    hydropathy_score = _score_hydropathy(sequence, constraints)
    charge_score = _score_charge(sequence, constraints)
    cysteine_penalty = _score_cysteines(sequence, constraints)
    repeat_penalty = _score_repeats(sequence, constraints)
    motif_bonus = _score_motifs(sequence)
    
    weighted_scores = {
        "hydropathy": reward_weights.get("physchem", 1.0) * hydropathy_score,
        "charge": reward_weights.get("physchem", 1.0) * charge_score,
        "cysteine_penalty": -reward_weights.get("physchem", 1.0) * cysteine_penalty,
        "repeat_penalty": -reward_weights.get("physchem", 1.0) * repeat_penalty,
        "motif_bonus": reward_weights.get("motif", 0.5) * motif_bonus
    }
    
    total_score = sum(weighted_scores.values())
    
    return {
        "total_score": total_score,
        "components": weighted_scores,
        "raw_metrics": {
            "hydropathy": calculate_hydropathy(sequence),
            "charge": calculate_charge(sequence),
            "cysteines": count_cysteines(sequence),
            "repeats": count_repeats(sequence),
            "motifs": contains_motif(sequence)
        }
    }
