"""
DNA Vocabulary class

This class handles converting DNA sequences into numbers that a neural network can understand.
The DNA "words" are called k-mers.

Example:
- DNA is made of 4 letters: A, C, G, T
- A k-mer is a sequence of k letters (ex: "ATCG" is a 4-mer)
- We will convert these letters into numbers for the NN
- This is similar to how NLP converts words into numbers (tokenization)
"""

from typing import List


class DNAVocabulary:
    """
    A simple vocabulary for DNA sequences.
    This class will create a "dictionary" that maps DNA k-mers to numbers.
    For example: "AAA" -> 0, "AAC" -> 1, "AAG" -> 2, etc.
    """

    def __init__(self, k: int = 6):
        """
        Initializing the vocabulary.

        It will take as arguments:
            k: The length of k-mers to use (default is 6)
                 Why 6?
               - Smaller k (3-4): fewer vocabulary items, less specific
               - Larger k (6-8): more vocabulary items, more specific
               - k=6 is chosen because most transcription factor binding motifs are 6-12 base pairs
        """
        self.k = k
        self.nucleotides = ['A', 'C', 'G', 'T']

        # Special tokens (like the punctuations in regular language)
        self.PAD = '<PAD>'  # Used to make all sequences the same length
        self.UNK = '<UNK>'  # Used for unknown/invalid k-mers
        self.CLS = '<CLS>'  # Used to mark the start of a sequence

        # Generating all possible k-mers
        print(f"Creating vocabulary with k={k}...")
        self.kmers = self._generate_all_kmers(k)

        # Creating the dictionaries for converting between k-mers and numbers
        # Forward mapping: k-mer -> number
        self.kmer_to_id = {}

        # Adding the special tokens first (they will get the ids: 0, 1, 2)
        self.kmer_to_id[self.PAD] = 0
        self.kmer_to_id[self.UNK] = 1
        self.kmer_to_id[self.CLS] = 2

        # Adding all k-mers (they will get the ids starting from 3)
        for idx, kmer in enumerate(self.kmers):
            self.kmer_to_id[kmer] = idx + 3

        # Reverse mapping: number -> k-mer
        self.id_to_kmer = {id_num: kmer for kmer, id_num in self.kmer_to_id.items()}

        # Storing vocabulary size
        self.vocab_size = len(self.kmer_to_id)

        print(f"Vocabulary created with {self.vocab_size} tokens")
        print(f"  - {len(self.kmers)} k-mers (4^{k} = {4 ** k})")
        print(f"  - 3 special tokens (PAD, UNK, CLS)")

    def _generate_all_kmers(self, k: int) -> List[str]:
        """
        Method that generates all possible k-mers.

        For example:
        For k=2, this creates: AA, AC, AG, AT, CA, CC, CG, CT, ...
        Total number of k-mers = 4^k

        It will take as argument: k, the length of k-mers

        It will return the list of all possible k-mers
        """
        if k == 1:
            return self.nucleotides

        # Starting with single nucleotides
        kmers = self.nucleotides.copy()

        # Building up to k-mers by adding one nucleotide at a time
        for _ in range(k - 1):
            new_kmers = []
            for kmer in kmers:
                for nucleotide in self.nucleotides:
                    new_kmers.append(kmer + nucleotide)
            kmers = new_kmers

        return kmers

    def sequence_to_kmers(self, sequence: str) -> List[str]:
        """
        Method that breaks a DNA sequence into k-mers using a sliding window.

        Example with k=3:
            Sequence: "ATCGATCG"
            K-mers: ["ATC", "TCG", "CGA", "GAT", "ATC", "TCG"]

        It will take as argument: DNA sequence string (ex: "ATCGATCG")

        It will return: List of k-mers
        """
        sequence = sequence.upper()  # Converting to uppercase
        kmers = []

        # Sliding a window of size k across the sequence
        # Stopping at position where we can still get a full k-mer
        for i in range(len(sequence) - self.k + 1):
            kmer = sequence[i:i + self.k]

            # Checking if this is a valid k-mer (it should only contain A, C, G, T)
            if all(nucleotide in self.nucleotides for nucleotide in kmer):
                kmers.append(kmer)
            else:
                # If invalid, using the unknown token
                kmers.append(self.UNK)

        return kmers

    def encode(self, sequence: str, max_length: int = None) -> List[int]:
        """
        Method that converts a DNA sequence into a list of numbers.

        This is what we feed into the neural network.

        Steps:
        1. Break sequence into k-mers
        2. Convert each k-mer to its ID number
        3. Add CLS token at the beginning
        4. Pad to max_length if specified

        It will take as arguments:
            sequence: DNA sequence string
            max_length: If specified, pad or truncate to this length

        It will return:
            List of integer IDs
        """
        # Step 1: Breaking into k-mers
        kmers = self.sequence_to_kmers(sequence)

        # Step 2: Converting to IDs
        ids = []
        for kmer in kmers:
            # Getting ID, or using UNK if not in vocabulary
            id_num = self.kmer_to_id.get(kmer, self.kmer_to_id[self.UNK])
            ids.append(id_num)

        # Step 3: Adding CLS token at the beginning
        ids = [self.kmer_to_id[self.CLS]] + ids

        # Step 4: Handling padding/truncation
        if max_length is not None:
            if len(ids) < max_length:
                # Padding with PAD tokens
                padding_needed = max_length - len(ids)
                ids = ids + [self.kmer_to_id[self.PAD]] * padding_needed
            elif len(ids) > max_length:
                # Truncating
                ids = ids[:max_length]

        return ids

    def decode(self, ids: List[int]) -> str:
        """
        Method that converts a list of IDs back into a DNA sequence.

        This is useful for debugging and visualization.

        It will take as argument:
            ids: List of integer IDs

        It will return:
            Reconstructed DNA sequence
        """
        kmers = []

        for id_num in ids:
            kmer = self.id_to_kmer.get(id_num, self.UNK)
            # Skipping special tokens
            if kmer not in [self.PAD, self.UNK, self.CLS]:
                kmers.append(kmer)

        return ''.join(kmers)
