"""
Integration Script: Combining Formal Language Methods with Transformer Model

This script demonstrates the full pipeline described in the abstract:
1. Regular expressions for motif detection
2. Context-Free Grammar for regulatory structure
3. Transformer model for TFBS prediction

It shows how formal language theory methods complement deep learning.
"""

import sys
import torch
from typing import Dict, List

# Import your existing components
try:
    from data.DNAVocabulary import DNAVocabulary
    from models.TransformerModel import TransformerModel
except ImportError:
    print("Note: Make sure DNAVocabulary and TransformerModel are in your path")

# Import the new formal language components
from motif_detector import MotifDetector
from regulatory_grammar import SimpleRegulatoryAnalyzer


class IntegratedDNAAnalyzer:
    """
    Integrated system combining formal language methods with deep learning.

    Pipeline:
    1. Regex → Quick motif detection (regular languages)
    2. CFG → Structural analysis (context-free languages)
    3. Transformer → Binding prediction (neural language model)

    This demonstrates the progression from simple to complex language processing.
    """

    def __init__(self,
                 model_path: str = None,
                 vocabulary: DNAVocabulary = None,
                 device: str = 'cpu'):
        """
        Initialize the integrated analyzer.

        Args:
            model_path: Path to trained transformer model (optional)
            vocabulary: DNA vocabulary for encoding (optional)
            device: Device to run model on
        """
        print("\n" + "=" * 70)
        print("INITIALIZING INTEGRATED DNA ANALYZER")
        print("=" * 70)

        # Component 1: Regex-based motif detector
        print("\n1. Loading Regex Motif Detector...")
        self.motif_detector = MotifDetector()

        # Component 2: CFG-based structural analyzer
        print("\n2. Loading CFG Regulatory Analyzer...")
        self.grammar_analyzer = SimpleRegulatoryAnalyzer()

        # Component 3: Transformer model (if available)
        print("\n3. Loading Transformer Model...")
        self.model = None
        self.vocabulary = vocabulary
        self.device = device

        if model_path and vocabulary:
            try:
                self.model = TransformerModel(
                    vocab_size=vocabulary.vocab_size,
                    d_model=128,
                    nhead=8,
                    num_layers=4
                )
                self.model.load_state_dict(torch.load(model_path, map_location=device))
                self.model.to(device)
                self.model.eval()
                print("   ✓ Transformer model loaded successfully")
            except Exception as e:
                print(f"   ⚠ Could not load transformer model: {e}")
        else:
            print("   ⚠ Transformer model not provided (using rule-based methods only)")

        print("\n" + "=" * 70)

    def analyze_sequence(self, sequence: str, name: str = "Unknown") -> Dict:
        """
        Comprehensive analysis using all three methods.

        Args:
            sequence: DNA sequence to analyze
            name: Name/ID for the sequence

        Returns:
            Dictionary with results from all three approaches
        """
        print(f"\n{'=' * 70}")
        print(f"ANALYZING SEQUENCE: {name}")
        print(f"{'=' * 70}")
        print(f"Length: {len(sequence)} bp")
        print(f"Sequence: {sequence[:60]}{'...' if len(sequence) > 60 else ''}")

        results = {
            'name': name,
            'sequence': sequence,
            'length': len(sequence)
        }

        # STEP 1: Regex-based motif detection
        print("\n" + "-" * 70)
        print("STEP 1: Regular Expression Analysis")
        print("-" * 70)
        print("Searching for known motif patterns...")

        motifs_found = self.motif_detector.find_motifs(sequence)
        motif_features = self.motif_detector.get_motif_features(sequence)

        results['regex_motifs'] = motifs_found
        results['motif_features'] = motif_features

        if motifs_found:
            print(f"✓ Found {len(motifs_found)} motif type(s):")
            for motif_name, matches in motifs_found.items():
                print(f"  • {motif_name}: {len(matches)} occurrence(s)")
                for pos, seq in matches[:2]:  # Show first 2
                    print(f"    - Position {pos}: {seq}")
        else:
            print("✗ No known motifs detected")

        # STEP 2: CFG-based structural analysis
        print("\n" + "-" * 70)
        print("STEP 2: Context-Free Grammar Analysis")
        print("-" * 70)
        print("Analyzing hierarchical regulatory structure...")

        structure_analysis = self.grammar_analyzer.analyze_regulatory_region(sequence)
        results['grammar_structure'] = structure_analysis

        print(f"Region Classification: {structure_analysis['region_type']}")
        print(f"Description: {structure_analysis['description']}")

        elements = [k for k, v in structure_analysis['core_elements'].items() if v]
        if elements:
            print(f"Structural Elements: {', '.join(elements)}")

        # STEP 3: Transformer prediction
        print("\n" + "-" * 70)
        print("STEP 3: Transformer Neural Language Model")
        print("-" * 70)

        if self.model and self.vocabulary:
            print("Predicting TFBS binding probability...")

            try:
                # Encode sequence
                encoded = self.vocabulary.encode(sequence, max_length=200)
                input_tensor = torch.tensor([encoded], dtype=torch.long).to(self.device)

                # Get prediction
                with torch.no_grad():
                    output = self.model(input_tensor).squeeze()
                    probability = torch.sigmoid(output).item()

                prediction = "BINDING" if probability > 0.5 else "NO BINDING"

                results['transformer_prediction'] = {
                    'probability': probability,
                    'prediction': prediction
                }

                print(f"✓ Prediction: {prediction}")
                print(f"  Binding probability: {probability:.4f}")

            except Exception as e:
                print(f"✗ Prediction failed: {e}")
                results['transformer_prediction'] = None
        else:
            print("⚠ Transformer model not available")
            results['transformer_prediction'] = None

        # SUMMARY: Combine insights
        print("\n" + "=" * 70)
        print("INTEGRATED ANALYSIS SUMMARY")
        print("=" * 70)

        self._print_integrated_summary(results)

        return results

    def _print_integrated_summary(self, results: Dict):
        """Print a summary combining all three approaches."""

        # Evidence from regex
        has_ctcf_motif = 'CTCF' in results['regex_motifs']
        motif_count = len(results['regex_motifs'])

        # Evidence from CFG
        is_regulatory = results['grammar_structure']['region_type'] != 'UNCLASSIFIED'
        region_type = results['grammar_structure']['region_type']

        # Evidence from transformer
        if results['transformer_prediction']:
            ml_prediction = results['transformer_prediction']['prediction']
            ml_confidence = results['transformer_prediction']['probability']
        else:
            ml_prediction = "N/A"
            ml_confidence = 0.0

        print("\nEvidence Summary:")
        print(f"  1. Regex Motifs:     {motif_count} known patterns found")
        if has_ctcf_motif:
            print(f"     → CTCF consensus motif detected!")

        print(f"  2. Grammar Analysis: Region type = {region_type}")
        if is_regulatory:
            print(f"     → Regulatory structure identified")

        print(f"  3. ML Prediction:    {ml_prediction}")
        if ml_prediction != "N/A":
            print(f"     → Confidence = {ml_confidence:.2%}")

        # Combined interpretation
        print("\nIntegrated Interpretation:")

        if has_ctcf_motif and ml_prediction == "BINDING":
            print("  ✓ STRONG EVIDENCE for CTCF binding")
            print("    - Consensus motif present (regex)")
            print("    - ML model predicts binding")
        elif ml_prediction == "BINDING" and is_regulatory:
            print("  ✓ LIKELY binding site")
            print("    - Regulatory structure present (CFG)")
            print("    - ML model predicts binding")
        elif ml_prediction == "NO BINDING" and motif_count == 0:
            print("  ✓ UNLIKELY to be binding site")
            print("    - No known motifs (regex)")
            print("    - ML model predicts no binding")
        else:
            print("  ? MIXED SIGNALS - further investigation needed")

        print("=" * 70)


def compare_approaches(sequences: List[tuple], analyzer: IntegratedDNAAnalyzer):
    """
    Compare how the three approaches perform on different sequences.

    This demonstrates the complementary strengths of:
    - Regex: Fast, interpretable, but limited to known patterns
    - CFG: Captures structure, but needs predefined grammar
    - Transformer: Learns patterns, but less interpretable
    """
    print("\n" + "=" * 70)
    print("COMPARING FORMAL LANGUAGE APPROACHES")
    print("=" * 70)

    print("\nApproach Characteristics:")
    print("  1. REGEX (Regular Languages)")
    print("     + Fast and efficient")
    print("     + Highly interpretable")
    print("     - Limited to linear patterns")
    print("     - Requires known motifs")

    print("\n  2. CFG (Context-Free Languages)")
    print("     + Models hierarchical structure")
    print("     + More expressive than regex")
    print("     - Requires explicit grammar")
    print("     - Less flexible than ML")

    print("\n  3. TRANSFORMER (Neural Language Model)")
    print("     + Learns patterns from data")
    print("     + Handles complex dependencies")
    print("     - Requires training data")
    print("     - Less interpretable")

    # Analyze each sequence
    for name, seq in sequences:
        analyzer.analyze_sequence(seq, name)
        print("\n" + "~" * 70 + "\n")


def main():
    """
    Main demonstration of integrated analysis.
    """
    print("\n" + "=" * 70)
    print("DNA-LM: INTEGRATED FORMAL LANGUAGE ANALYSIS")
    print("=" * 70)
    print("\nDemonstrating the progression:")
    print("  Regex → CFG → Transformer")
    print("  (Regular Languages → Context-Free Languages → Neural Language Models)")
    print("=" * 70)

    # Initialize analyzer
    # Note: For full functionality, provide model_path and vocabulary
    analyzer = IntegratedDNAAnalyzer()

    # Test sequences (designed to show different characteristics)
    test_sequences = [
        ("Strong CTCF site",
         "ATCGCCGCGTAGGAGGCAGGGGCGGTATAAATGCTAGC"),  # Has CTCF motif + regulatory elements

        ("Promoter region",
         "GCTAGCTATAAAGGCTAGCTATGCGATCG"),  # Has TATA box + ATG

        ("Enhancer region",
         "ATCGGGGCGGTAGCTAGCTAGGCCAATCT"),  # Has GC box + CAAT box

        ("Random sequence",
         "ATATATCGCGATCGATCGATATATCGCG")  # No known motifs
    ]

    # Run comparison
    compare_approaches(test_sequences, analyzer)

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nKey Takeaway:")
    print("  This project demonstrates how formal language theory")
    print("  (regex, CFG) provides interpretable foundations while")
    print("  neural language models (transformers) add learned")
    print("  pattern recognition for complex biological sequences.")
    print("=" * 70)


if __name__ == "__main__":
    main()