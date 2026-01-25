"""
DNA Vocabulary class

This class handles converting DNA sequences into numbers that a neural network can understand.
The DNA "words" are called k-mers.

Example:
- DNA is made of 4 letters: A, C, G, T
- A k-mer is a sequence of k letters (ex: "ATCG" is a 4-mer)
- We convert these letters into numbers for the NN
- This is similar to how NLP converts words into numbers (tokenization)

Why is this needed?
Neural networks can't read letters (A, T, C, G) - they only understand numbers.
We need to translate DNA into a numerical format, just like translating English to French.

Key concepts:
- K-mer: A "word" made of k DNA letters (ex: k=6 → "ATCGAT" is a 6-mer)
- Vocabulary: A dictionary mapping k-mers to numbers (ex: "AAA" → 0, "AAC" → 1)
- Tokenization: The process of converting text into numbers
- Special tokens: Special markers like <CLS> (start), <PAD> (fill empty space)
"""

from typing import List


class DNAVocabulary:
    """
    A simple vocabulary for DNA sequences.

    What does this class do?
    It creates a "dictionary" that maps DNA k-mers to numbers.
    For example: "AAA" -> 0, "AAC" -> 1, "AAG" -> 2, etc.

    Analogy
    - English dictionary: Maps words to definitions
    - DNA vocabulary: Maps k-mers to ID numbers

    Why do we need this?
    The transformer model can only process numbers, not letters.
    This class is the translator between DNA (ATCG) and numbers (0, 1, 2, 3...).
    """

    def __init__(self, k: int = 6):
        """
        Initializing the vocabulary.

        What happens here:
        1. We will choose a k-mer size (how many letters per "word")
        2. We will generate ALL possible k-mers (ex: for k=2 → AA, AC, AG, AT, CA, CC...)
        3. We will assign each k-mer a unique ID number
        4. We will create two dictionaries:
           - Forward: k-mer → number (for encoding DNA)
           - Backward: number → k-mer (for decoding back to DNA)

        Parameters used:
            k: The length of k-mers to use (default is 6)

               Why 6?
               - k=6 is chosen because most transcription factor binding motifs are 6-12 base pairs
                 * Our 6-mers can capture the core motif patterns
                 * Not too large (manageable vocabulary size)
                 * Not too small (specific enough for biology)

        What gets created:
        - self.k: Stores the k value (6 by default)
        - self.kmers: List of all possible k-mers
        - self.kmer_to_id: Dictionary mapping k-mer → number
        - self.id_to_kmer: Dictionary mapping number → k-mer
        - self.vocab_size: Total number of tokens (k-mers + special tokens)
        """
        # Storing k-mer size
        self.k = k

        # The 4 building blocks of DNA (nucleotides)
        self.nucleotides = ['A', 'C', 'G', 'T']
        # PAD token: Used to make all sequences the same length
        self.PAD = '<PAD>'

        # UNK token: Used for unknown/invalid k-mers
        self.UNK = '<UNK>'

        # CLS token: Used to mark the start of a sequence
        self.CLS = '<CLS>'

        # Generating all possible k-mers
        # For k=6, this creates 4^6 = 4,096 k-mers
        print(f"Creating vocabulary with k={k}...")
        self.kmers = self._generate_all_kmers(k)

        # Creating the dictionaries for converting between k-mers and numbers

        # Forward mapping: k-mer -> number
        self.kmer_to_id = {}

        self.kmer_to_id[self.PAD] = 0
        self.kmer_to_id[self.UNK] = 1
        self.kmer_to_id[self.CLS] = 2

        # Adding all k-mers (they get IDs starting from 3)
        for idx, kmer in enumerate(self.kmers):
            self.kmer_to_id[kmer] = idx + 3

        # Reverse mapping: number -> k-mer
        self.id_to_kmer = {id_num: kmer for kmer, id_num in self.kmer_to_id.items()}

        # Storing vocabulary size (total number of tokens)
        # This will be: 4^k + 3 (k-mers + special tokens)
        # For k=6: 4,096 + 3 = 4,099 tokens
        self.vocab_size = len(self.kmer_to_id)

        print(f"Vocabulary created with {self.vocab_size} tokens")
        print(f"  - {len(self.kmers)} k-mers (4^{k} = {4 ** k})")
        print(f"  - 3 special tokens (PAD, UNK, CLS)")

    def _generate_all_kmers(self, k: int) -> List[str]:
        """
        Method that generates all possible k-mers.
        """
        # Base case: if k=1, just return the nucleotides
        if k == 1:
            return self.nucleotides

        kmers = self.nucleotides.copy()

        # Building up to k-mers by adding one nucleotide at a time
        for _ in range(k - 1):
            # Creating a new list to store the longer k-mers
            new_kmers = []

            for kmer in kmers:
                for nucleotide in self.nucleotides:
                    # Example: "AT" + "C" = "ATC"
                    new_kmers.append(kmer + nucleotide)

            kmers = new_kmers

        return kmers

    def sequence_to_kmers(self, sequence: str) -> List[str]:
        """
        Method that breaks a DNA sequence into k-mers using a sliding window.

        What does this do?
        It takes a long DNA sequence and chops it into overlapping k-mers.

        Why overlapping?
        Because binding motifs can start at any position in the sequence.
        We don't want to miss a motif just because we started chopping at the wrong place.

        Parameters used:
            sequence: DNA sequence string (ex: "ATCGATCG")

        It will return:
            List of k-mers

        Example with k=6:
            Input:  "ATCGATCGATCG" (12 bases)
            Output: ["ATCGAT", "TCGATC", "CGATCG", "GATCGA", "ATCGAT", "TCGATC", "CGATCG"]
                    (7 k-mers from a 12-base sequence)

        Formula for number of k-mers:
            Number of k-mers = sequence_length - k + 1
            Example: 12 - 6 + 1 = 7 k-mers
        """
        # Converting to uppercase to ensure consistency
        sequence = sequence.upper()
        kmers = []

        # Sliding a window of size k across the sequence
        for i in range(len(sequence) - self.k + 1):
            # Extracting a k-mer starting at position i
            # [i:i+k] means "from position i to position i+k (not included)"
            # Example: if i=2, k=3, then [2:5] gives characters at positions 2,3,4
            kmer = sequence[i:i + self.k]

            # Checking if this is a valid k-mer (only contains A, C, G, T)
            if all(nucleotide in self.nucleotides for nucleotide in kmer):
                # Valid k-mer, adding it to our list
                kmers.append(kmer)
            else:
                # Invalid k-mer, using the unknown token
                kmers.append(self.UNK)

        return kmers

    def encode(self, sequence: str, max_length: int = None) -> List[int]:
        """
        Method that converts a DNA sequence into a list of numbers.
        This is what we feed into the neural network.
        What happens step by step:
        1. Breaking sequence into k-mers
           "ATCGATCG" → ["ATCGAT", "TCGATC", "CGATCG"]

        2. Converting each k-mer to its ID number
           ["ATCGAT", "TCGATC", "CGATCG"] → [42, 137, 203]

        3. Adding CLS token at the beginning
           [42, 137, 203] → [2, 42, 137, 203]
           (Remember: CLS has ID=2)

        4. Padding or truncating to max_length if specified
           [2, 42, 137, 203] → [2, 42, 137, 203, 0, 0, 0, ...]
           (Padding with PAD tokens which have ID=0)

        Parameters used:
            sequence: DNA sequence string
            max_length: If specified, pad or truncate to this length
                       If None, return the natural length

        It will return:
            List of integer IDs that the model can process
        """

        # Step 1: Breaking into k-mers
        kmers = self.sequence_to_kmers(sequence)

        # Step 2: Converting to IDs
        # For each k-mer, looking up its ID number in our vocabulary
        ids = []
        for kmer in kmers:
            id_num = self.kmer_to_id.get(kmer, self.kmer_to_id[self.UNK])
            ids.append(id_num)

        # Step 3: Adding CLS token at the beginning
        ids = [self.kmer_to_id[self.CLS]] + ids

        # Step 4: Handling padding/truncation
        if max_length is not None:
            if len(ids) < max_length:
                padding_needed = max_length - len(ids)
                ids = ids + [self.kmer_to_id[self.PAD]] * padding_needed

            elif len(ids) > max_length:
                ids = ids[:max_length]

        return ids

    def decode(self, ids: List[int]) -> str:
        """
        Method that converts a list of IDs back into a DNA sequence.

        What does this do?
        It reverses the encoding process.
        Numbers → k-mers → DNA sequence

        How it works:
        1. For each ID number, looking up its k-mer
        2. Skipping special tokens (PAD, UNK, CLS)
        3. Joining all k-mers into one string

        Parameters used:
            ids: List of integer IDs (ex: [2, 42, 137, 203, 0, 0])

        It will return:
            Reconstructed DNA sequence (approximate)
        """
        kmers = []

        # For each ID in the input list
        for id_num in ids:
            # Looking up what k-mer this ID corresponds to
            kmer = self.id_to_kmer.get(id_num, self.UNK)

            # Skipping special tokens
            if kmer not in [self.PAD, self.UNK, self.CLS]:
                kmers.append(kmer)

        # Joining all k-mers into one continuous string
        return ''.join(kmers)