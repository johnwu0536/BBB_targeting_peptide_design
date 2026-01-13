"""
Amino acid utilities for L1 baseline.
Handles 20 canonical amino acids and their properties.
"""

# 20 canonical amino acids
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

# Kyte-Doolittle hydropathy scale (normalized)
HYDROPATHY = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
}

# Charge properties
POSITIVE_AAS = ['K', 'R', 'H']
NEGATIVE_AAS = ['D', 'E']

# Hydrophobic amino acids
HYDROPHOBIC_AAS = ['A', 'C', 'F', 'I', 'L', 'M', 'V', 'W']

# Common motifs for BBB penetration
BBB_MOTIFS = [
    "RGD",  # Cell adhesion motif
    "NGR",  # Tumor homing motif  
    "CPP",  # Cell penetrating peptide patterns
    "TAT",  # HIV-TAT derived
    "PEN",  # Penetratin
]


def calculate_hydropathy(sequence: str) -> float:
    """Calculate average hydropathy score for a sequence."""
    if not sequence:
        return 0.0
    return sum(HYDROPATHY[aa] for aa in sequence) / len(sequence)


def calculate_charge(sequence: str) -> int:
    """Calculate net charge of a sequence."""
    positive = sum(1 for aa in sequence if aa in POSITIVE_AAS)
    negative = sum(1 for aa in sequence if aa in NEGATIVE_AAS)
    return positive - negative


def count_cysteines(sequence: str) -> int:
    """Count number of cysteine residues."""
    return sequence.count('C')


def count_repeats(sequence: str, min_repeat: int = 2) -> int:
    """Count number of repeating amino acid patterns."""
    repeats = 0
    i = 0
    while i < len(sequence):
        aa = sequence[i]
        count = 1
        while i + count < len(sequence) and sequence[i + count] == aa:
            count += 1
        if count >= min_repeat:
            repeats += count - 1  # Count excess beyond single occurrence
        i += count
    return repeats


def contains_motif(sequence: str, motifs: list = None) -> int:
    """Check if sequence contains any of the specified motifs."""
    if motifs is None:
        motifs = BBB_MOTIFS
    
    motif_count = 0
    for motif in motifs:
        if motif in sequence:
            motif_count += 1
    
    return motif_count


def validate_sequence(sequence: str, min_len: int = 8, max_len: int = 20) -> bool:
    """Validate sequence length and composition."""
    if len(sequence) < min_len or len(sequence) > max_len:
        return False
    
    if not all(aa in AA_ALPHABET for aa in sequence):
        return False
    
    return True


def generate_random_sequence(length: int) -> str:
    """Generate a random peptide sequence."""
    import random
    return ''.join(random.choice(AA_ALPHABET) for _ in range(length))


def mutate_sequence(sequence: str, mutation_rate: float = 0.1) -> str:
    """Mutate a sequence by randomly changing amino acids."""
    import random
    
    mutated = list(sequence)
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Choose a different amino acid
            current_aa = mutated[i]
            possible_aas = [aa for aa in AA_ALPHABET if aa != current_aa]
            mutated[i] = random.choice(possible_aas)
    
    return ''.join(mutated)


def crossover_sequences(seq1: str, seq2: str) -> str:
    """Perform single-point crossover between two sequences."""
    import random
    
    min_len = min(len(seq1), len(seq2))
    crossover_point = random.randint(1, min_len - 1)
    
    # Choose which parent contributes which part
    if random.random() < 0.5:
        return seq1[:crossover_point] + seq2[crossover_point:]
    else:
        return seq2[:crossover_point] + seq1[crossover_point:]


def get_sequence_stats(sequence: str) -> dict:
    """Get comprehensive statistics for a sequence."""
    return {
        'length': len(sequence),
        'hydropathy': calculate_hydropathy(sequence),
        'charge': calculate_charge(sequence),
        'cysteines': count_cysteines(sequence),
        'repeats': count_repeats(sequence),
        'motifs': contains_motif(sequence),
        'hydrophobic_ratio': sum(1 for aa in sequence if aa in HYDROPHOBIC_AAS) / len(sequence)
    }
