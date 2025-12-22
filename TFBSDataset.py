import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List
import DNAVocabulary as DNAVocabulary

class TFBSDataset(Dataset):
    """
    Dataset for Transcription Factor Binding Site prediction.

    Demonstrates bioinformatics concepts:
    - DNA sequence representation
    - Reverse complement augmentation
    - Label encoding for binding sites
    """

    def __init__(self, sequences: List[str], labels: List[int],
                 vocabulary: DNAVocabulary, max_length: int = 200,
                 augment: bool = True):
        """
        Initialize TFBS dataset.

        Args:
            sequences: List of DNA sequences
            labels: Binary labels (1 = binding site, 0 = non-binding)
            vocabulary: DNAVocabulary instance
            max_length: Maximum sequence length
            augment: Whether to use reverse complement augmentation
        """
        self.sequences = sequences
        self.labels = labels
        self.vocabulary = vocabulary
        self.max_length = max_length
        self.augment = augment

    def __len__(self):
        return len(self.sequences)

    def reverse_complement(self, sequence: str) -> str:
        """
        Generate reverse complement of DNA sequence.

        Biologically motivated augmentation:
        DNA is double-stranded, so binding can occur on either strand.
        """
        complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
        return ''.join([complement.get(base, base) for base in sequence[::-1]])

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]

        # Random augmentation: use reverse complement 50% of the time
        if self.augment and np.random.random() > 0.5:
            sequence = self.reverse_complement(sequence)

        # Encode sequence
        token_ids = self.vocabulary.encode(sequence, max_length=self.max_length)

        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.float),
            'sequence': sequence  # For interpretability analysis
        }