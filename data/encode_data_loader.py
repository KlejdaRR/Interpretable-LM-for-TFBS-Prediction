
"""
ENCODE Data Loader

This file helps us load real ChIP-seq data from the ENCODE project and prepares it for training.

What is ENCODE?
ENCODE = Encyclopedia of DNA Elements
It's a public database of genomic experiments from thousands of labs worldwide.

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
    print(f"\n{'= ' *70}")
    print("LOADING ENCODE CHIP-SEQ DATA")
    print(f"{'= ' *70}")
    print(f"File: {peaks_file}")
    print(f"Target: CTCF transcription factor")
    print(f"Loading up to {max_sequences} peaks...")

    sequences = []  # DNA sequences (strings like "ATCGATCG...")
    labels = []     # Labels (1 for binding, 0 for non-binding)
    skipped = 0     # Counter for sequences we had to skip

    try:
        if peaks_file.endswith('.gz'):
            file_handle = gzip.open(peaks_file, 'rt')
        else:
            file_handle = open(peaks_file, 'r')

        with file_handle as f:
            for i, line in enumerate(f):
                if len(sequences) >= max_sequences:
                    break

                if line.startswith('#') or line.startswith('track') or line.startswith('browser'):
                    continue

                parts = line.strip().split('\t')

                if len(parts) < 3:
                    continue

                chrom = parts[0]      # Chromosome (ex: "chr1", "chr2", "chrX")
                start = int(parts[1]) # Start position (ex: 123456)
                end = int(parts[2])   # End position (ex: 123756)

                # Skipping non-standard chromosomes
                standard_chroms = [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY']
                if chrom not in standard_chroms:
                    continue

                # Calculating center of peak and extracting 200bp window
                center = (start + end) // 2
                seq_start = max(0, center - 100)
                seq_end = center + 100

                try:
                    url = "https://api.genome.ucsc.edu/getData/sequence"

                    params = {
                        'genome': 'hg38',
                        'chrom': chrom,
                        'start': seq_start,
                        'end': seq_end
                    }

                    response = requests.get(url, params=params, timeout=10)

                    if response.status_code == 200:
                        data = response.json()
                        seq = data['dna'].upper()

                        # Quality control: only keeping valid sequences
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

    print(f"\n Successfully loaded {len(sequences)} positive sequences")
    print(f"  (Skipped {skipped} sequences due to quality control)")

    # Generating negative examples (non-binding sites)
    print("\nGenerating negative examples...")
    print("  (Random genomic sequences as controls)")

    neg_sequences = []
    # Creating the same number of negatives as positives (balanced dataset)
    # This prevents bias (model learning to just predict the majority class)
    for i in range(len(sequences)):
        seq = ''.join(random.choices(['A', 'C', 'G', 'T'], k=200))
        neg_sequences.append(seq)

        if (i + 1) % 200 == 0:
            print(f"  Generated {i + 1}/{len(sequences)} negative sequences...")

    neg_labels = [0] * len(neg_sequences)
    all_sequences = sequences + neg_sequences
    all_labels = labels + neg_labels

    # Shuffling the data
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
    """
    if data_path is None:
        print("\nGenerating synthetic data for demonstration...")

        def generate_random_sequence(length=200):
            return ''.join(random.choices(['A', 'C', 'G', 'T'], k=length))

        n_samples = 1000
        sequences = [generate_random_sequence() for _ in range(n_samples)]

        # Generating random labels (50/50 split on average)
        labels = [random.randint(0, 1) for _ in range(n_samples)]

        print(f"Generated {n_samples} synthetic sequences")
        print(f"  - Positive (binding): {sum(labels)}")  # Counting 1s
        print(f"  - Negative (no binding): {n_samples - sum(labels)}")  # Counting 0s

        return sequences, labels
    else:
        return load_encode_peaks(data_path, max_sequences=1000)