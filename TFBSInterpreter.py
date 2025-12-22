import torch
from typing import List
from collections import Counter
import TransformerTFBSPredictor as TransformerTFBSPredictor
import DNAVocabulary as DNAVocabulary

class TFBSInterpreter:
    """
    Interpretability tools for biological insight extraction.

    Demonstrates integration of:
    - Attention visualization
    - Motif discovery
    - Biological validation
    """

    def __init__(self, model: TransformerTFBSPredictor, vocabulary: DNAVocabulary):
        self.model = model
        self.vocabulary = vocabulary
        self.model.eval()

    def extract_attention_scores(self, sequence: str, max_length: int = 200):
        """
        Extract attention scores for a given sequence.
        """
        # Encode sequence
        token_ids = self.vocabulary.encode(sequence, max_length=max_length)
        input_ids = torch.tensor([token_ids])

        with torch.no_grad():
            # This is a placeholder - would need to modify transformer
            # to actually return attention weights
            output = self.model(input_ids)

        return output

    def find_motifs(self, sequences: List[str], labels: List[int],
                    min_length: int = 6, max_length: int = 12):
        """
        Extract candidate motifs from sequences with positive predictions.

        This could be enhanced with:
        - Statistical significance testing
        - Comparison to JASPAR database
        - Motif logo generation
        """
        positive_sequences = [seq for seq, label in zip(sequences, labels) if label == 1]

        # Simple k-mer counting approach
        motif_counts = Counter()

        for seq in positive_sequences:
            for k in range(min_length, max_length + 1):
                for i in range(len(seq) - k + 1):
                    kmer = seq[i:i + k]
                    if all(base in self.vocabulary.alphabet for base in kmer):
                        motif_counts[kmer] += 1

        # Return top motifs
        top_motifs = motif_counts.most_common(20)
        return top_motifs


# ============================================================================
# EXAMPLE USAGE AND DEMONSTRATION
# ============================================================================

def demonstrate_dna_lm():
    """
    Demonstration of the complete DNA-LM pipeline.
    """
    print("=" * 80)
    print("DNA-LM: Interpretable Language Model for TFBS Prediction")
    print("=" * 80)

    # 1. Create vocabulary (demonstrates formal language concepts)
    print("\n1. Creating DNA Vocabulary (k-mer tokenization)")
    vocab = DNAVocabulary(k=6)

    # Example tokenization
    example_seq = "ATCGATCGATCGATCG"
    tokens = vocab.tokenize(example_seq)
    print(f"Example sequence: {example_seq}")
    print(f"Tokens: {tokens}")
    print(f"Encoded: {vocab.encode(example_seq)}")

    # 2. Create model
    print("\n2. Creating Hybrid CNN-Transformer Model")
    model = TransformerTFBSPredictor(
        vocab_size=vocab.vocab_size,
        embedding_dim=128,
        num_heads=8,
        num_layers=4
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 3. Demonstrate architecture components
    print("\n3. Model Architecture:")
    print(model)

    print("\n" + "=" * 80)
    print("Project demonstrates integration of:")
    print("- Language Processing: Formal languages, tokenization, transformers")
    print("- Machine Learning: Deep networks, attention, optimization")
    print("- Bioinformatics: DNA sequences, TFBS prediction, motif discovery")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_dna_lm()