"""
Enhanced main script that demonstrates the full language processing hierarchy:
Regex → CFG → Transformer for DNA sequence analysis

This version better integrates all language processing technologies as described
in the project description.
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import random
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from data.DNAVocabulary import DNAVocabulary
from data.TFBSDataset import TFBSDataset
from models.TransformerModel import TransformerModel
from training.Trainer import Trainer
from visualization.AttentionVisualizer import AttentionVisualizer
from LPT_Implementation import RegexMotifDetector
from LPT_Implementation import DNAGrammar


def demonstrate_language_hierarchy(example_sequences):
    """
    Demonstrate the complete language processing hierarchy:
    Type 3 (Regex) → Type 2 (CFG) → Beyond (Transformer)
    """
    print("\n" + "=" * 80)
    print("FORMAL LANGUAGE HIERARCHY IN DNA ANALYSIS")
    print("Demonstrating progression from simple to complex language tools")
    print("=" * 80)

    # LEVEL 1: Regular Expressions (Type 3 - Chomsky Hierarchy)
    print("\n" + "🔹" * 20 + " LEVEL 1: REGULAR EXPRESSIONS " + "🔹" * 20)
    print("Type 3 Languages - What they can do: Simple pattern matching")
    print("Type 3 Languages - What they cannot do: Nested structures, context")

    regex_detector = RegexMotifDetector()

    for i, seq in enumerate(example_sequences[:2], 1):
        print(f"\nExample {i}: {seq[:60]}...")
        analysis = regex_detector.analyze_sequence(seq, verbose=True)

    # LEVEL 2: Context-Free Grammars (Type 2 - Chomsky Hierarchy)
    print("\n" + "🔸" * 20 + " LEVEL 2: CONTEXT-FREE GRAMMARS " + "🔸" * 20)
    print("Type 2 Languages - What they can do: Hierarchical structure, composition")
    print("Type 2 Languages - What they cannot do: Context-sensitive dependencies")

    dna_grammar = DNAGrammar()

    # Demonstrate CFG with example structures
    grammar_examples = [
        "TATA CAAT TSS",  # Simple promoter
        "CTCF EBOX CTCF",  # Simple enhancer
        "TATA SPACER CAAT SPACER GC SPACER TSS"  # Complex promoter
    ]

    for example in grammar_examples:
        print(f"\nTesting structure: {example}")
        is_valid = dna_grammar.validate_structure(example)
        print(f"Valid regulatory structure: {is_valid}")
        if is_valid:
            dna_grammar.analyze_structure(example, verbose=False)

    # LEVEL 3: Transformer Neural Networks (Beyond Chomsky)
    print("\n" + "🔶" * 15 + " LEVEL 3: TRANSFORMER NEURAL NETWORKS " + "🔶" * 15)
    print("Beyond formal languages - What they can do: Learn ANY pattern from data")
    print("Capabilities: Long-range dependencies, context-sensitive patterns, combinations")

    print("\nTransformer capabilities that exceed formal grammars:")
    print("  Position-dependent binding (same motif, different contexts)")
    print("  Long-range interactions (position 10 affects position 150)")
    print("  Learned patterns (discovers motifs not explicitly programmed)")
    print("  Quantitative predictions (0.85 binding probability)")
    print("  Attention-based interpretability")


def compare_approaches(sequences, labels, transformer_model, vocabulary):
    """
    Compare all three language processing approaches on the same data
    """
    print("\n" + "=" * 80)
    print("COMPARATIVE ANALYSIS: REGEX vs CFG vs TRANSFORMER")
    print("=" * 80)

    regex_detector = RegexMotifDetector()
    dna_grammar = DNAGrammar()

    correct_regex = 0
    correct_transformer = 0
    total_evaluated = 0

    print("\nAnalyzing sample sequences with all three approaches...")

    for i in range(min(10, len(sequences))):
        seq = sequences[i]
        true_label = labels[i]

        print(f"\n{'─' * 60}")
        print(f"Sequence {i + 1}: {seq[:40]}... (True: {'Binding' if true_label else 'No binding'})")

        # REGEX APPROACH
        regex_analysis = regex_detector.analyze_sequence(seq, verbose=False)
        regex_prediction = 1 if regex_analysis['has_CTCF'] else 0
        regex_correct = (regex_prediction == true_label)

        # TRANSFORMER APPROACH
        encoded = vocabulary.encode(seq, max_length=200)
        input_tensor = torch.tensor([encoded], dtype=torch.long)

        with torch.no_grad():
            transformer_output = transformer_model(input_tensor)
            transformer_prob = torch.sigmoid(transformer_output).item()
            transformer_prediction = 1 if transformer_prob > 0.5 else 0
            transformer_correct = (transformer_prediction == true_label)

        print(f"  Regex:       {regex_prediction} ({'✓' if regex_correct else '✗'})")
        print(
            f"  Transformer: {transformer_prediction} ({transformer_prob:.3f}) ({'✓' if transformer_correct else '✗'})")

        if regex_correct:
            correct_regex += 1
        if transformer_correct:
            correct_transformer += 1
        total_evaluated += 1

    print(f"\n{'=' * 60}")
    print("SUMMARY COMPARISON:")
    print(f"Regex Accuracy:       {correct_regex / total_evaluated:.3f} ({correct_regex}/{total_evaluated})")
    print(
        f"Transformer Accuracy: {correct_transformer / total_evaluated:.3f} ({correct_transformer}/{total_evaluated})")
    print(f"{'=' * 60}")

    return {
        'regex_accuracy': correct_regex / total_evaluated,
        'transformer_accuracy': correct_transformer / total_evaluated
    }


def demonstrate_tokenization_biology(vocabulary):
    """
    Demonstrate how k-mer tokenization relates to biological meaning
    """
    print("\n" + "=" * 80)
    print("TOKENIZATION: FROM DNA TO BIOLOGICALLY MEANINGFUL UNITS")
    print("=" * 80)

    example_seq = "ATCGATCGTATAATAAGCGGGCGGCTCAG"

    print(f"Original DNA sequence: {example_seq}")
    print(f"Length: {len(example_seq)} base pairs")

    print(f"\nTokenization with k={vocabulary.k}:")
    kmers = vocabulary.sequence_to_kmers(example_seq)
    print(f"K-mers: {kmers}")
    print(f"Number of k-mers: {len(kmers)}")

    print(f"\nEncoding to numbers:")
    encoded = vocabulary.encode(example_seq)
    print(f"Encoded: {encoded[:10]}... (first 10 tokens)")

    print(f"\nBiological relevance of k-mer choice (k={vocabulary.k}):")
    print(f"  • Most TF binding motifs are 6-12 bp long")
    print(f"  • K=6 captures core motif patterns")
    print(f"  • Vocabulary size: 4^6 + special tokens = {vocabulary.vocab_size}")
    print(f"  • Sliding window preserves all possible binding sites")


def main():
    """
    Enhanced main function demonstrating the full language processing hierarchy
    """
    print("\n" + "=" * 80)
    print("DNA AS FORMAL LANGUAGE: COMPLETE LANGUAGE PROCESSING PIPELINE")
    print("Alphabet Σ = {A, T, C, G} | Task: Transcription Factor Binding Site Prediction")
    print("=" * 80)

    # Configuration
    config = {
        'data_path': None,  # Use synthetic data for demo
        'max_seq_length': 200,
        'k': 6,
        'd_model': 128,
        'nhead': 8,
        'num_layers': 4,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'num_epochs': 5,  # Reduced for demo
        'output_dir': './outputs',
        'random_seed': 42,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # Set seed and create output directory
    random.seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    torch.manual_seed(config['random_seed'])
    os.makedirs(config['output_dir'], exist_ok=True)

    # Generate sample data
    print("\nGenerating synthetic DNA sequences for demonstration...")

    def generate_random_sequence(length=200):
        return ''.join(random.choices(['A', 'C', 'G', 'T'], k=length))

    sequences = [generate_random_sequence() for _ in range(200)]
    labels = [random.randint(0, 1) for _ in range(200)]

    # Create vocabulary (tokenizer)
    print("\nCreating DNA vocabulary (k-mer tokenizer)...")
    vocabulary = DNAVocabulary(k=config['k'])

    # DEMONSTRATION 1: Language Hierarchy
    demonstrate_language_hierarchy(sequences)

    # DEMONSTRATION 2: Tokenization
    demonstrate_tokenization_biology(vocabulary)

    # DEMONSTRATION 3: Quick transformer training
    print("\n" + "=" * 80)
    print("TRAINING TRANSFORMER MODEL (Brief Demo)")
    print("=" * 80)

    # Create small dataset for quick demo
    train_dataset = TFBSDataset(
        sequences=sequences[:150],
        labels=labels[:150],
        vocabulary=vocabulary,
        max_length=config['max_seq_length']
    )
    val_dataset = TFBSDataset(
        sequences=sequences[150:],
        labels=labels[150:],
        vocabulary=vocabulary,
        max_length=config['max_seq_length']
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    # Create and train transformer
    transformer_model = TransformerModel(
        vocab_size=vocabulary.vocab_size,
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        max_seq_length=config['max_seq_length']
    )

    trainer = Trainer(
        model=transformer_model,
        device=config['device'],
        learning_rate=config['learning_rate']
    )

    # Quick training
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['num_epochs'],
        early_stopping_patience=3,
        save_dir=config['output_dir']
    )

    # DEMONSTRATION 4: Comparative Analysis
    comparison_results = compare_approaches(
        sequences[150:], labels[150:],
        transformer_model, vocabulary
    )

    # DEMONSTRATION 5: Attention Visualization
    print("\n" + "=" * 80)
    print("INTERPRETABILITY: ATTENTION VISUALIZATION")
    print("=" * 80)

    visualizer = AttentionVisualizer(transformer_model, vocabulary)

    # Visualize one example
    example_seq = sequences[150]
    print(f"\nAnalyzing sequence: {example_seq[:50]}...")

    attention_data = visualizer.get_attention_weights(example_seq)
    print(f"Transformer prediction: {attention_data['prediction']:.3f}")

    # Save visualization
    heatmap_path = os.path.join(config['output_dir'], 'demo_attention_heatmap.png')
    visualizer.plot_attention_heatmap(attention_data, save_path=heatmap_path)

    # Final summary
    print("\n" + "=" * 80)
    print("LANGUAGE PROCESSING TECHNOLOGIES DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nSummary of implemented technologies:")
    print("  Regular Expressions (Type 3) - Pattern matching for known motifs")
    print("  Context-Free Grammars (Type 2) - Hierarchical regulatory structures")
    print("  K-mer Tokenization - Biologically meaningful segmentation")
    print("  Transformer Architecture - Neural language model for TFBS prediction")
    print("  Attention Visualization - Interpretability of learned patterns")
    print(f"\nAll outputs saved in: {config['output_dir']}")
    print("=" * 80)


if __name__ == "__main__":
    main()