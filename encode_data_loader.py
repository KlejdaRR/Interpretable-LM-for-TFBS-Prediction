"""
ENCODE Data Loader
==================

This script loads real ENCODE ChIP-seq data and prepares it for training.
"""

import gzip
import requests
import time
import random
from typing import List, Tuple


def load_encode_peaks(peaks_file: str, max_sequences: int = 1000) -> Tuple[List[str], List[int]]:
    """
    Load ENCODE ChIP-seq peaks and fetch DNA sequences.

    Args:
        peaks_file: Path to downloaded .bed.gz file (e.g., ENCFF308JDD.bed.gz)
        max_sequences: Maximum number of sequences to load

    Returns:
        (sequences, labels) tuple where sequences are DNA strings and labels are 0/1
    """
    print(f"\n{'=' * 70}")
    print("LOADING ENCODE CHIP-SEQ DATA")
    print(f"{'=' * 70}")
    print(f"File: {peaks_file}")
    print(f"Target: CTCF transcription factor")
    print(f"Loading up to {max_sequences} peaks...")

    sequences = []
    labels = []
    skipped = 0

    # Open the gzipped BED file
    try:
        with gzip.open(peaks_file, 'rt') as f:
            for i, line in enumerate(f):
                # Stop when we have enough sequences
                if len(sequences) >= max_sequences:
                    break

                # Skip header/comment lines
                if line.startswith('#') or line.startswith('track') or line.startswith('browser'):
                    continue

                # Parse BED format: chrom start end name score ...
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue

                chrom = parts[0]
                start = int(parts[1])
                end = int(parts[2])

                # Skip non-standard chromosomes
                if chrom not in [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY']:
                    continue

                # Calculate center of peak and extract 200bp window
                center = (start + end) // 2
                seq_start = max(0, center - 100)  # 100bp on each side = 200bp total
                seq_end = center + 100

                # Fetch sequence from UCSC Genome Browser API
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

                        # Quality control: only keep valid sequences
                        if len(seq) == 200 and 'N' not in seq:
                            sequences.append(seq)
                            labels.append(1)  # 1 = binding site

                            # Progress update
                            if len(sequences) % 100 == 0:
                                print(f"  Loaded {len(sequences)}/{max_sequences} sequences...")
                        else:
                            skipped += 1
                    else:
                        skipped += 1

                    # Be nice to the API (rate limiting)
                    time.sleep(0.15)

                except Exception as e:
                    skipped += 1
                    if len(sequences) % 100 == 0:
                        print(f"  Warning: Skipped {skipped} peaks due to errors")
                    continue

    except FileNotFoundError:
        print(f"\n ERROR: File not found: {peaks_file}")
        print("Please download the file first!")
        return [], []
    except Exception as e:
        print(f"\n ERROR loading file: {e}")
        return [], []

    print(f"\n✓ Successfully loaded {len(sequences)} positive sequences")
    print(f"  (Skipped {skipped} sequences due to quality control)")

    # Generate negative examples (non-binding sites)
    print("\nGenerating negative examples...")
    print("  (Random genomic sequences as controls)")

    neg_sequences = []
    for i in range(len(sequences)):
        # Generate random DNA sequence
        seq = ''.join(random.choices(['A', 'C', 'G', 'T'], k=200))
        neg_sequences.append(seq)

        if (i + 1) % 200 == 0:
            print(f"  Generated {i + 1}/{len(sequences)} negative sequences...")

    neg_labels = [0] * len(neg_sequences)

    # Combine positive and negative examples
    all_sequences = sequences + neg_sequences
    all_labels = labels + neg_labels

    # Shuffle the data
    combined = list(zip(all_sequences, all_labels))
    random.shuffle(combined)
    all_sequences, all_labels = zip(*combined)

    print(f"\n{'=' * 70}")
    print("DATASET SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total sequences: {len(all_sequences)}")
    print(f"  - Positive (CTCF binding): {len(sequences)} ({len(sequences) / len(all_sequences) * 100:.1f}%)")
    print(f"  - Negative (no binding):   {len(neg_sequences)} ({len(neg_sequences) / len(all_sequences) * 100:.1f}%)")
    print(f"Sequence length: 200 bp")
    print(f"{'=' * 70}\n")

    return list(all_sequences), list(all_labels)


def load_data(data_path: str = None):
    """
    Main data loading function.

    Args:
        data_path: Path to ENCODE peaks file, or None for synthetic data

    Returns:
        (sequences, labels) tuple
    """
    if data_path is None:
        # Generate synthetic data for testing
        print("\nGenerating synthetic data for demonstration...")

        def generate_random_sequence(length=200):
            return ''.join(random.choices(['A', 'C', 'G', 'T'], k=length))

        n_samples = 1000
        sequences = [generate_random_sequence() for _ in range(n_samples)]
        labels = [random.randint(0, 1) for _ in range(n_samples)]

        print(f"Generated {n_samples} synthetic sequences")
        print(f"  - Positive (binding): {sum(labels)}")
        print(f"  - Negative (no binding): {n_samples - sum(labels)}")

        return sequences, labels
    else:
        # Load real ENCODE data
        return load_encode_peaks(data_path, max_sequences=1000)

