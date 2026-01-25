"""
This class handles loading and preparing DNA sequence data for training.
- training data (sequences + labels)
- Each item has an input (DNA sequence) and output (0 or 1: binding or not)
- Data augmentation: creating variations of data to help the model learn better
- For DNA, we will use "reverse complement" - the opposite strand has the same biological meaning
"""

import torch
from torch.utils.data import Dataset
import random
from typing import List


class TFBSDataset(Dataset):
    """
    A dataset for Transcription Factor Binding Site prediction.

    What is a TFBS?
    Transcription factors (TFs) are proteins that bind to specific DNA sequences
    to control gene expression. We want to predict WHERE they bind.

    Dataset Structure:
    - Input: DNA sequence (ex: "ATCGATCG...")
    - Output: Label (1 = TF binds here, 0 = TF doesn't bind)
    """

    def __init__(self,
                 sequences: List[str],
                 labels: List[int],
                 vocabulary,
                 max_length: int = 200,
                 use_augmentation: bool = True):
        """
        Initializing the dataset.

        It will take as arguments:
            sequences: List of DNA sequences as strings
            labels: List of binary labels (1 = binding site, 0 = no binding)
            vocabulary: DNAVocabulary class instance for encoding
            max_length: Maximum sequence length (sequences will be padded/truncated)
            use_augmentation: Whether to use reverse complement augmentation
        """
        self.sequences = sequences
        self.labels = labels
        self.vocabulary = vocabulary
        self.max_length = max_length
        self.use_augmentation = use_augmentation

        print(f"\nDataset created:")
        print(f"  - {len(sequences)} sequences")
        print(f"  - Positive samples (binding): {sum(labels)}")
        print(f"  - Negative samples (no binding): {len(labels) - sum(labels)}")
        print(f"  - Max sequence length: {max_length}")
        print(f"  - Data augmentation: {'ON' if use_augmentation else 'OFF'}")

    def __len__(self):
        return len(self.sequences)

    def reverse_complement(self, sequence: str) -> str:
        """
        Method that generates the reverse complement of a DNA sequence.

        Why are we doing this?
        Because DNA is double-stranded. If a TF binds to one strand, the sequence on
        the other strand is the reverse complement. Both are biologically equivalent.

        Example:
            Original:  5' - A T C G - 3'
                           | | | |
            Complement: 3' - T A G C - 5'

            Reverse complement: 5' - C G A T - 3'

        Rules:
            A ↔ T
            C ↔ G

        It will take as arguments:
            sequence: DNA sequence string

        It will return:
            Reverse complement sequence
        """
        # Defining complementary base pairs
        complement_map = {
            'A': 'T',
            'T': 'A',
            'C': 'G',
            'G': 'C'
        }

        # Step 1: Reversing the sequence
        reversed_seq = sequence[::-1]

        # Step 2: Getting the complement of each base
        rev_comp = ''.join([complement_map.get(base, base) for base in reversed_seq])

        return rev_comp

    def __getitem__(self, idx):
        """
        Constructor method that will get a single item from the dataset.

        It will take as arguments:
            idx: Index of the item to retrieve

        It will return:
            Dictionary with:
                - 'input_ids': Encoded sequence (list of numbers)
                - 'label': Binding label (0 or 1)
                - 'sequence': Original DNA sequence (for debugging)
        """
        sequence = self.sequences[idx]
        label = self.labels[idx]

        # Data augmentation: randomly using reverse complement
        # This will help the model learn that both strands are equivalent
        if self.use_augmentation and random.random() > 0.5:
            sequence = self.reverse_complement(sequence)

        # Encoding the sequence to numbers
        encoded_sequence = self.vocabulary.encode(sequence, max_length=self.max_length)

        # Converting to PyTorch tensors
        return {
            'input_ids': torch.tensor(encoded_sequence, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float),
            'sequence': sequence
        }
