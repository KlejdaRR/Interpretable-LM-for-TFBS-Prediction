from typing import List, Optional


class DNAVocabulary:
    """
    Vocabulary class implementing formal language concepts for DNA sequences.

    Demonstrates LPT concepts:
    - Alphabet: Σ = {A, C, G, T}
    - Language: L ⊆ Σ*
    - Tokenization: k-mer based (analogous to BPE)
    """

    def __init__(self, k: int = 6):
        """
        Initialize DNA vocabulary with k-mer tokenization.

        Args:
            k: Length of k-mers (default 6 for biological relevance)
        """
        self.k = k
        self.alphabet = ['A', 'C', 'G', 'T']

        # Generate all possible k-mers (vocabulary)
        self.vocab = self._generate_kmers(k)
        self.token2id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id2token = {idx: token for token, idx in self.token2id.items()}

        # Special tokens (following NLP conventions)
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.cls_token = '<CLS>'  # For classification tasks

        # Add special tokens to vocabulary
        special_tokens = [self.pad_token, self.unk_token, self.cls_token]
        for token in special_tokens:
            if token not in self.token2id:
                idx = len(self.vocab)
                self.vocab.append(token)
                self.token2id[token] = idx
                self.id2token[idx] = token

        self.vocab_size = len(self.vocab)
        print(f"Vocabulary created with {self.vocab_size} tokens (k={k})")

    def _generate_kmers(self, k: int) -> List[str]:
        """
        Generate all possible k-mers from DNA alphabet.

        This creates the vocabulary V from the formal language L over Σ.
        For k=3: AAA, AAC, AAG, AAT, ACA, ... (4^k tokens)
        """
        if k == 1:
            return self.alphabet

        kmers = []

        def generate(current, remaining):
            if remaining == 0:
                kmers.append(current)
                return
            for nucleotide in self.alphabet:
                generate(current + nucleotide, remaining - 1)

        generate('', k)
        return kmers

    def tokenize(self, sequence: str, stride: int = None) -> List[str]:
        """
        Tokenize DNA sequence into k-mers.

        Implements sliding window tokenization (analogous to word tokenization in NLP).

        Args:
            sequence: DNA sequence string
            stride: Step size for sliding window (default: k, non-overlapping)

        Returns:
            List of k-mer tokens
        """
        if stride is None:
            stride = self.k

        sequence = sequence.upper()
        tokens = []

        for i in range(0, len(sequence) - self.k + 1, stride):
            kmer = sequence[i:i + self.k]
            if all(nucleotide in self.alphabet for nucleotide in kmer):
                tokens.append(kmer)
            else:
                tokens.append(self.unk_token)

        return tokens

    def encode(self, sequence: str, max_length: Optional[int] = None) -> List[int]:
        """
        Encode DNA sequence to token IDs.

        Args:
            sequence: DNA sequence string
            max_length: Maximum sequence length (with padding)

        Returns:
            List of token IDs
        """
        tokens = self.tokenize(sequence)
        token_ids = [self.token2id.get(token, self.token2id[self.unk_token])
                     for token in tokens]

        # Add CLS token at the beginning (for classification)
        token_ids = [self.token2id[self.cls_token]] + token_ids

        # Pad if necessary
        if max_length:
            if len(token_ids) < max_length:
                padding = [self.token2id[self.pad_token]] * (max_length - len(token_ids))
                token_ids = token_ids + padding
            else:
                token_ids = token_ids[:max_length]

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to DNA sequence.
        """
        tokens = [self.id2token.get(idx, self.unk_token) for idx in token_ids]
        # Remove special tokens
        tokens = [t for t in tokens if t not in [self.pad_token, self.unk_token, self.cls_token]]
        return ''.join(tokens)