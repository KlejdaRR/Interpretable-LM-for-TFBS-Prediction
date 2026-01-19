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
               - Smaller k (3-4): Fewer vocabulary items, less specific
                 * k=3 → 4^3 = 64 possible k-mers (small vocabulary)
                 * Less computational cost, but misses important patterns

               - Larger k (6-8): More vocabulary items, more specific
                 * k=6 → 4^6 = 4,096 possible k-mers (medium vocabulary)
                 * Better captures biological motifs

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

        # Special tokens (like punctuation in regular language)
        # These are not DNA k-mers but serve special purposes

        # PAD token: Used to make all sequences the same length
        # Think: Padding sentences with spaces to align them
        # Example: "hello" → "hello     " (padded to 10 characters)
        self.PAD = '<PAD>'

        # UNK token: Used for unknown/invalid k-mers
        # Think: A word you don't know, so you replace it with [UNKNOWN]
        # Example: If DNA has "N" (ambiguous base), we use <UNK>
        self.UNK = '<UNK>'

        # CLS token: Used to mark the start of a sequence
        # Think: Capital letter at the start of a sentence
        # Example: "<CLS> ATCGAT TCGATC CGATCG" (CLS marks the beginning)
        self.CLS = '<CLS>'

        # Generating all possible k-mers
        # For k=6, this creates 4^6 = 4,096 k-mers
        print(f"Creating vocabulary with k={k}...")
        self.kmers = self._generate_all_kmers(k)

        # Creating the dictionaries for converting between k-mers and numbers

        # Forward mapping: k-mer -> number
        # This is used when ENCODING DNA → numbers
        self.kmer_to_id = {}

        # Adding the special tokens first (they get the first IDs: 0, 1, 2)
        # Why first? So they have consistent IDs across all vocabularies
        self.kmer_to_id[self.PAD] = 0  # Padding token gets ID 0
        self.kmer_to_id[self.UNK] = 1  # Unknown token gets ID 1
        self.kmer_to_id[self.CLS] = 2  # Classification token gets ID 2

        # Adding all k-mers (they get IDs starting from 3)
        # enumerate() gives us: (0, "AAA"), (1, "AAC"), (2, "AAG"), ...
        # We add 3 to skip the special tokens: 0+3=3, 1+3=4, 2+3=5, ...
        for idx, kmer in enumerate(self.kmers):
            self.kmer_to_id[kmer] = idx + 3

        # Reverse mapping: number -> k-mer
        # This is used when DECODING numbers → DNA
        # We flip the dictionary: {kmer: id} → {id: kmer}
        self.id_to_kmer = {id_num: kmer for kmer, id_num in self.kmer_to_id.items()}

        # Storing vocabulary size (total number of tokens)
        # This will be: 4^k + 3 (k-mers + special tokens)
        # For k=6: 4,096 + 3 = 4,099 tokens
        self.vocab_size = len(self.kmer_to_id)

        # Printing summary
        print(f"Vocabulary created with {self.vocab_size} tokens")
        print(f"  - {len(self.kmers)} k-mers (4^{k} = {4 ** k})")
        print(f"  - 3 special tokens (PAD, UNK, CLS)")

    def _generate_all_kmers(self, k: int) -> List[str]:
        """
        Method that generates all possible k-mers.

        What does this do?
        It creates every possible combination of k DNA letters.

        For example:
        - k=1: Creates ["A", "C", "G", "T"] (4 k-mers)
        - k=2: Creates ["AA", "AC", "AG", "AT", "CA", "CC", ...] (16 k-mers)
        - k=3: Creates ["AAA", "AAC", "AAG", "AAT", "ACA", ...] (64 k-mers)
        - k=6: Creates 4,096 k-mers

        Total number of k-mers = 4^k (4 choices for each position, k positions)

        How it works:
        We start with single letters [A, C, G, T]
        Then we add another letter to each: [AA, AC, AG, AT, CA, CC, CG, CT, ...]
        Then we add another letter to each: [AAA, AAC, AAG, AAT, ACA, ...]
        We repeat this k-1 times to get k-mers

        Parameters used:
            k: The length of k-mers to generate

        It will return:
            List of all possible k-mers

        Example with k=2:
        Start:  ["A", "C", "G", "T"]
        Round 1: For each letter, add A, C, G, T
                ["AA", "AC", "AG", "AT",  ← from "A"
                 "CA", "CC", "CG", "CT",  ← from "C"
                 "GA", "GC", "GG", "GT",  ← from "G"
                 "TA", "TC", "TG", "TT"]  ← from "T"
        Result: 16 2-mers
        """
        # Base case: if k=1, just return the nucleotides
        # This is our starting point
        if k == 1:
            return self.nucleotides

        # Starting with single nucleotides
        # This is like starting with the alphabet before making words
        kmers = self.nucleotides.copy()

        # Building up to k-mers by adding one nucleotide at a time
        # We do this k-1 times (we already have 1-mers, need to grow to k-mers)
        for _ in range(k - 1):
            # Creating a new list to store the longer k-mers
            new_kmers = []

            # For each existing k-mer...
            for kmer in kmers:
                # ...add each possible nucleotide to the end
                for nucleotide in self.nucleotides:
                    # Example: "AT" + "C" = "ATC"
                    new_kmers.append(kmer + nucleotide)

            # Replacing old k-mers with the new, longer ones
            # Example: After round 1, we go from ["A", "C", "G", "T"] to ["AA", "AC", ...]
            kmers = new_kmers

        return kmers

    def sequence_to_kmers(self, sequence: str) -> List[str]:
        """
        Method that breaks a DNA sequence into k-mers using a sliding window.

        What does this do?
        It takes a long DNA sequence and chops it into overlapping k-mers.

        Analogy:
        Imagine reading a sentence 3 letters at a time, sliding one letter forward each time:
        "HELLO" with window size 3:
        - Position 0: "HEL"
        - Position 1: "ELL"
        - Position 2: "LLO"

        For DNA with k=3:
        "ATCGATCG" becomes:
        - Position 0: "ATC"
        - Position 1: "TCG"
        - Position 2: "CGA"
        - Position 3: "GAT"
        - Position 4: "ATC"
        - Position 5: "TCG"

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
        # "atcg" → "ATCG" (neural networks need consistent input)
        sequence = sequence.upper()

        # List to store the k-mers we extract
        kmers = []

        # Sliding a window of size k across the sequence
        # We stop when we can't fit a full k-mer anymore
        #
        # Example with k=3, sequence="ATCGAT" (length 6):
        # - i=0: "ATC" (positions 0-2) ✓
        # - i=1: "TCG" (positions 1-3) ✓
        # - i=2: "CGA" (positions 2-4) ✓
        # - i=3: "GAT" (positions 3-5) ✓
        # - i=4: would need positions 4-6, but position 6 doesn't exist ✗
        #
        # So we stop at: len(sequence) - k + 1 = 6 - 3 + 1 = 4 positions (0,1,2,3)
        for i in range(len(sequence) - self.k + 1):
            # Extracting a k-mer starting at position i
            # [i:i+k] means "from position i to position i+k (not included)"
            # Example: if i=2, k=3, then [2:5] gives characters at positions 2,3,4
            kmer = sequence[i:i + self.k]

            # Checking if this is a valid k-mer (only contains A, C, G, T)
            # Why check? Sometimes DNA sequences contain:
            # - "N" = ambiguous base (could be any nucleotide)
            # - "-" = gap
            # - Other characters from sequencing errors
            #
            # all() returns True only if ALL nucleotides are valid
            if all(nucleotide in self.nucleotides for nucleotide in kmer):
                # Valid k-mer, adding it to our list
                kmers.append(kmer)
            else:
                # Invalid k-mer, using the unknown token
                # This tells the model "I don't know what this is"
                kmers.append(self.UNK)

        return kmers

    def encode(self, sequence: str, max_length: int = None) -> List[int]:
        """
        Method that converts a DNA sequence into a list of numbers.

        This is what we feed into the neural network.
        The model can't read "ATCG", but it can process [42, 137, 203, 512].

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

        Why do we need padding?
        Neural networks process data in batches (multiple sequences at once).
        All sequences in a batch must have the same length.
        Shorter sequences are padded with <PAD> tokens.

        Example:
        Sequence 1: "ATCG" → encoded length 5
        Sequence 2: "ATCGATCGATCG" → encoded length 13
        After padding to max_length=20:
        Sequence 1: [2, 42, 137, 203, 512, 0, 0, 0, ...] (length 20)
        Sequence 2: [2, 42, 137, 203, 512, 693, ...] (length 20)

        Parameters used:
            sequence: DNA sequence string
            max_length: If specified, pad or truncate to this length
                       If None, return the natural length

        It will return:
            List of integer IDs that the model can process

        Real example with k=6:
            Input: "ATCGATCGATCG", max_length=10

            Step 1: Breaking into k-mers
            → ["ATCGAT", "TCGATC", "CGATCG", "GATCGA", "ATCGAT", "TCGATC", "CGATCG"]

            Step 2: Converting to IDs (these are made-up IDs for illustration)
            → [42, 137, 203, 512, 42, 137, 203]

            Step 3: Adding CLS token
            → [2, 42, 137, 203, 512, 42, 137, 203]

            Step 4: Truncating to max_length=10
            → [2, 42, 137, 203, 512, 42, 137, 203, 0, 0]
            (We had 8, needed 10, so added 2 PAD tokens)
        """
        # Step 1: Breaking into k-mers
        # This splits the long sequence into overlapping windows
        kmers = self.sequence_to_kmers(sequence)

        # Step 2: Converting to IDs
        # For each k-mer, looking up its ID number in our vocabulary
        ids = []
        for kmer in kmers:
            # Getting ID from the dictionary
            # .get(kmer, default) means: "get the ID for this k-mer,
            #                             or use the UNK ID if not found"
            # This handles invalid k-mers gracefully
            id_num = self.kmer_to_id.get(kmer, self.kmer_to_id[self.UNK])
            ids.append(id_num)

        # Step 3: Adding CLS token at the beginning
        # The CLS token marks "this is where the sequence starts"
        # [id1, id2, id3] → [CLS_ID, id1, id2, id3]
        # [42, 137, 203] → [2, 42, 137, 203]
        ids = [self.kmer_to_id[self.CLS]] + ids

        # Step 4: Handling padding/truncation
        # Making sure the sequence has exactly max_length tokens
        if max_length is not None:
            # Case 1: Sequence is too short → add padding
            if len(ids) < max_length:
                # Calculating how many PAD tokens we need
                padding_needed = max_length - len(ids)
                # Adding PAD tokens (ID=0) at the end
                # [2, 42, 137] + [0, 0, 0, 0, 0] = [2, 42, 137, 0, 0, 0, 0, 0]
                ids = ids + [self.kmer_to_id[self.PAD]] * padding_needed

            # Case 2: Sequence is too long → truncate
            elif len(ids) > max_length:
                # Cutting off the end to fit max_length
                # [2, 42, 137, 203, 512, 693] → [2, 42, 137, 203, 512] (if max_length=5)
                ids = ids[:max_length]

        return ids

    def decode(self, ids: List[int]) -> str:
        """
        Method that converts a list of IDs back into a DNA sequence.

        What does this do?
        It reverses the encoding process.
        Numbers → k-mers → DNA sequence

        This is useful for:
        - Debugging: Checking what the model actually saw
        - Visualization: Showing sequences in a readable format
        - Understanding: Verifying that encoding/decoding works correctly

        How it works:
        1. For each ID number, looking up its k-mer
        2. Skipping special tokens (PAD, UNK, CLS)
        3. Joining all k-mers into one string

        Parameters used:
            ids: List of integer IDs (ex: [2, 42, 137, 203, 0, 0])

        It will return:
            Reconstructed DNA sequence (approximate)

        Note: The reconstruction is approximate because k-mers overlap!
        Original:   "ATCGATCG"
        K-mers:     ["ATCGAT", "TCGATC", "CGATCG"]
        Joined:     "ATCGATTCGATCCGATCG" (wrong! overlaps weren't handled)

        For perfect reconstruction, you'd need to handle overlaps properly.
        But for visualization purposes, this simple joining often works fine.

        Example:
            Input IDs:  [2, 42, 137, 203, 0, 0]

            Step 1: Looking up k-mers
            2  → "<CLS>" (special token, skip)
            42 → "ATCGAT"
            137 → "TCGATC"
            203 → "CGATCG"
            0  → "<PAD>" (special token, skip)
            0  → "<PAD>" (special token, skip)

            Step 2: Joining valid k-mers
            → "ATCGATTCGATCCGATCG"
        """
        # List to store k-mers we decode
        kmers = []

        # For each ID in the input list
        for id_num in ids:
            # Looking up what k-mer this ID corresponds to
            # .get(id_num, default) returns the k-mer or UNK if ID not found
            kmer = self.id_to_kmer.get(id_num, self.UNK)

            # Skipping special tokens
            # We only want the actual DNA k-mers, not <PAD>, <UNK>, or <CLS>
            if kmer not in [self.PAD, self.UNK, self.CLS]:
                kmers.append(kmer)

        # Joining all k-mers into one continuous string
        # ["ATCGAT", "TCGATC"] → "ATCGATTCGATC"
        return ''.join(kmers)