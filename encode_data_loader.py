
"""
ENCODE Data Loader

This file helps us load real ChIP-seq data from the ENCODE project and prepares it for training.

What is ENCODE?
ENCODE = Encyclopedia of DNA Elements
It's a public database of genomic experiments from thousands of labs worldwide.
We can download real experimental data showing where transcription factors bind DNA.

What is ChIP-seq?
ChIP-seq = Chromatin Immunoprecipitation sequencing
It's an experimental technique that finds where proteins bind to DNA.
Steps: 1) Cells are treated to freeze protein-DNA interactions
       2) DNA is broken into fragments
       3) Antibodies pull out fragments bound to our protein (e.g., CTCF)
       4) Sequencing tells us where those fragments came from in the genome
       5) Result: A list of genomic coordinates where the protein binds

What do we get?
A .bed file (Browser Extensible Data) containing:
- Chromosome: Which chromosome (chr1, chr2, ..., chrX, chrY)
- Start: Where the binding region starts (base pair position)
- End: Where the binding region ends (base pair position)
- Peak: The strongest binding point in that region

Example line from .bed file:
chr1    123456    123756    peak_1    1000    .    5.2    10.3    8.4    150
This means: "CTCF binds on chromosome 1, from position 123,456 to 123,756"
"""

import gzip
import requests
import time
import random
from typing import List, Tuple


def load_encode_peaks(peaks_file: str, max_sequences: int = 1000) -> Tuple[List[str], List[int]]:
    """
    Loading ENCODE ChIP-seq peaks and fetching DNA sequences.

    What does this do?
    1. Reading the .bed file containing binding site coordinates
    2. For each binding site, calculating the center point
    3. Extracting 200bp of DNA sequence around that center (±100bp)
    4. Fetching the actual DNA sequence from UCSC Genome Browser database
    5. Generating negative examples (random sequences as controls)
    6. Combining and shuffling positive and negative examples

    Why 200bp?
    - CTCF binding motifs are typically 15-20 bp
    - We include surrounding context (±100bp) for two reasons:
      1. Flanking regions may contain co-factors or regulatory elements
      2. The model can learn position-invariant features
    - 200bp is long enough for context but short enough to process efficiently

    What are positive and negative examples?
    - Positive (label=1): Real CTCF binding sites from ChIP-seq experiments
      These are places where CTCF actually binds in living cells

    - Negative (label=0): Random genomic sequences where CTCF doesn't bind
      We generate these artificially as controls
      Why random? True negatives are hard to define (just because ChIP-seq
      didn't find binding doesn't mean it never happens)

    Parameters used:
        peaks_file: Path to downloaded .bed or .bed.gz file
                   Example: "./ENCFF308JDD.bed.gz"
        max_sequences: Maximum number of sequences to load
                      Default: 1000 (good for testing)
                      More sequences = better model but slower training

    It will return:
        Tuple of (sequences, labels) where:
        - sequences: List of DNA strings (200bp each)
        - labels: List of 0s and 1s (0=no binding, 1=binding)

    Example return:
        sequences = ["ATCGATCG...", "GCTAGCTA...", ...]  (2000 sequences total)
        labels = [1, 0, 1, 1, 0, ...]  (1000 ones, 1000 zeros)
    """
    # Printing header information
    print(f"\n{'= ' *70}")
    print("LOADING ENCODE CHIP-SEQ DATA")
    print(f"{'= ' *70}")
    print(f"File: {peaks_file}")
    print(f"Target: CTCF transcription factor")
    print(f"Loading up to {max_sequences} peaks...")

    # Lists to store our results
    sequences = []  # DNA sequences (strings like "ATCGATCG...")
    labels = []     # Labels (1 for binding, 0 for non-binding)
    skipped = 0     # Counter for sequences we had to skip

    try:
        # Opening the file
        if peaks_file.endswith('.gz'):
            file_handle = gzip.open(peaks_file, 'rt')
        else:
            file_handle = open(peaks_file, 'r')

        # Reading the file line by line
        with file_handle as f:
            for i, line in enumerate(f):
                # Stopping when we have enough sequences
                if len(sequences) >= max_sequences:
                    break

                # Skipping header lines
                if line.startswith('#') or line.startswith('track') or line.startswith('browser'):
                    continue

                parts = line.strip().split('\t')

                # Quality check: make sure we have at least 3 columns
                # We need: chromosome, start, end
                if len(parts) < 3:
                    continue

                # Extracting the coordinates
                chrom = parts[0]      # Chromosome (ex: "chr1", "chr2", "chrX")
                start = int(parts[1]) # Start position (ex: 123456)
                end = int(parts[2])   # End position (ex: 123756)

                # Skipping non-standard chromosomes
                # We only want the main chromosomes: 1-22, X, Y
                # Why skip others? They might be:
                # - Mitochondrial DNA (chrM)
                # - Unplaced sequences (chr1_random)
                # - Patches or alternative sequences
                # These are less reliable and complicate the analysis
                standard_chroms = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY']
                if chrom not in standard_chroms:
                    continue

                # Calculating center of peak and extracting 200bp window
                # Peak region: start=123456, end=123756 (300bp wide)
                # Center: (123456 + 123756) / 2 = 123606
                # Window: center ± 100bp = 123506 to 123706 (200bp total)
                center = (start + end) // 2  # // is integer division (no decimals)
                seq_start = max(0, center - 100)  # 100bp before center (but not negative)
                seq_end = center + 100            # 100bp after center

                # Fetching sequence from UCSC Genome Browser API
                # We have coordinates (chr1:123506-123706) but not the actual DNA sequence
                # UCSC provides an API to download sequences by coordinate
                try:
                    url = "https://api.genome.ucsc.edu/getData/sequence"

                    params = {
                        'genome': 'hg38',      # Human genome version 38 (latest standard)
                        'chrom': chrom,        # Which chromosome (ex: chr1)
                        'start': seq_start,    # Start position
                        'end': seq_end         # End position
                    }

                    response = requests.get(url, params=params, timeout=10)

                    # Checking if request was successful
                    if response.status_code == 200:
                        data = response.json()
                        seq = data['dna'].upper()  # Getting DNA and converting to uppercase

                        # Quality control: only keeping valid sequences
                        # We check two things:
                        # 1. Length is exactly 200bp (no truncation at chromosome ends)
                        # 2. No 'N' characters (N means "unknown nucleotide" from sequencing)
                        if len(seq) == 200 and 'N' not in seq:
                            sequences.append(seq)
                            labels.append(1)  # 1 = positive example (binding site)

                            if len(sequences) % 100 == 0:
                                print(f"  Loaded {len(sequences)}/{max_sequences} sequences...")
                        else:
                            skipped += 1
                    else:
                        skipped += 1

                    time.sleep(0.15)

                except Exception as e:
                    skipped += 1

                    if len(sequences) % 100 == 0:
                        print(f"  Warning: Skipped {skipped} peaks due to errors")

                    continue

    except FileNotFoundError:
        print(f"\nERROR: File not found: {peaks_file}")
        print("Please download the file first!")
        return [], []

    except Exception as e:
        print(f"\nERROR loading file: {e}")
        return [], []

    print(f"\n✓ Successfully loaded {len(sequences)} positive sequences")
    print(f"  (Skipped {skipped} sequences due to quality control)")

    # Generating negative examples (non-binding sites)
    # Why do we need negatives?
    # The model needs to learn the difference between:
    # - Sequences where CTCF binds (positives)
    # - Sequences where CTCF doesn't bind (negatives)
    # Without negatives, the model would just say "binding!" for everything
    print("\nGenerating negative examples...")
    print("  (Random genomic sequences as controls)")

    neg_sequences = []
    # Creating the same number of negatives as positives (balanced dataset)
    # Balanced means: 50% positive, 50% negative
    # This prevents bias (model learning to just predict the majority class)
    for i in range(len(sequences)):
        # Generating random DNA sequence
        # random.choices picks k random items from the list
        # Example: random.choices(['A','C','G','T'], k=5) → ['A','T','C','G','A']
        # Then ''.join() combines them into a string: "ATCGA"
        seq = ''.join(random.choices(['A', 'C', 'G', 'T'], k=200))
        neg_sequences.append(seq)

        if (i + 1) % 200 == 0:
            print(f"  Generated {i + 1}/{len(sequences)} negative sequences...")

    # Creating labels for negative sequences (all zeros)
    # [0] * 5 creates [0, 0, 0, 0, 0]
    neg_labels = [0] * len(neg_sequences)

    # Combining positive and negative examples
    # We now have:
    # - 1000 positive sequences (real binding sites) with label 1
    # - 1000 negative sequences (random DNA) with label 0
    # Total: 2000 sequences
    all_sequences = sequences + neg_sequences      # Concatenating lists
    all_labels = labels + neg_labels               # Concatenating labels

    # Shuffling the data
    # Why shuffle? We don't want all positives first, then all negatives
    # That would confuse the training (model sees patterns in order)
    combined = list(zip(all_sequences, all_labels))
    random.shuffle(combined)
    all_sequences, all_labels = zip(*combined)

    print(f"\n{'= ' *70}")
    print("DATASET SUMMARY")
    print(f"{'= ' *70}")
    print(f"Total sequences: {len(all_sequences)}")
    print(f"  - Positive (CTCF binding): {len(sequences)} ({len(sequences ) /len(all_sequences ) *100:.1f}%)")
    print(f"  - Negative (no binding):   {len(neg_sequences)} ({len(neg_sequences ) /len(all_sequences ) *100:.1f}%)")
    print(f"Sequence length: 200 bp")
    print(f"{'= ' *70}\n")

    return list(all_sequences), list(all_labels)


def load_data(data_path: str = None):
    """
    Main data loading function that chooses between real and synthetic data.

    What does this do?
    It's a wrapper function that decides:
    - If data_path is provided → load real ENCODE data
    - If data_path is None → generate synthetic random data (for testing)

    Why have synthetic data?
    1. Testing: Quickly test if code works without downloading files
    2. Debugging: Synthetic data has known properties (50/50 split, no errors)
    3. Demonstration: Show the pipeline without requiring ENCODE access

    Note: Synthetic data gives poor results (43% accuracy)
    This is EXPECTED because random DNA has no learnable patterns!
    Real ENCODE data gives good results (97.8% accuracy)

    Parameters used:
        data_path: Path to ENCODE peaks file
                  If None, generates synthetic data instead
                  Example: "./ENCFF308JDD.bed.gz"

    It will return:
        Tuple of (sequences, labels) where:
        - sequences: List of DNA strings
        - labels: List of 0s and 1s

    Example usage:
        # Loading real data
        seqs, labels = load_data("./ENCFF308JDD.bed.gz")

        # Loading synthetic data (for testing)
        seqs, labels = load_data(None)
    """
    # Checking if we should generate synthetic data
    if data_path is None:
        # Generating synthetic data for testing
        print("\nGenerating synthetic data for demonstration...")

        # Helper function to create random DNA sequence
        def generate_random_sequence(length=200):
            # Creating a random string of A, C, G, T nucleotides
            # This has NO biological meaning - purely random
            return ''.join(random.choices(['A', 'C', 'G', 'T'], k=length))

        # Generating 1000 random sequences
        n_samples = 1000
        sequences = [generate_random_sequence() for _ in range(n_samples)]

        # Generating random labels (50/50 split on average)
        # random.randint(0, 1) picks either 0 or 1 randomly
        labels = [random.randint(0, 1) for _ in range(n_samples)]

        print(f"Generated {n_samples} synthetic sequences")
        print(f"  - Positive (binding): {sum(labels)}")  # Counting 1s
        print(f"  - Negative (no binding): {n_samples - sum(labels)}")  # Counting 0s

        return sequences, labels
    else:
        # Loading real ENCODE data
        # Calling the load_encode_peaks function with our file path
        return load_encode_peaks(data_path, max_sequences=1000)