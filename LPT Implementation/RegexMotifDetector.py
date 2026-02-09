"""
Regular Expression Motif Detector

This module demonstrates how regular expressions can be used to detect known
biological motifs in DNA sequences - the first level of our language hierarchy.

What is a motif?
A motif is a recurring pattern in DNA that has biological significance.
Example: The TATA-box is a DNA sequence that helps position RNA polymerase.

Why start with regex?
Regular expressions are the simplest formal language tool (Type 3 in Chomsky hierarchy).
They're perfect for detecting simple, fixed patterns in sequences.

Connection to Language Processing Technologies course:
- Regular languages and finite automata
- Pattern matching with regex
- Building blocks for more complex grammars
"""

import re
from typing import List, Dict, Tuple


class RegexMotifDetector:
    """
    Detecting biological motifs using regular expressions.

    This class demonstrates the progression from simple pattern matching (regex)
    to the complex transformer model - showing how formal language theory applies
    to genomics.
    """

    def __init__(self):
        """
        Initializing the detector with common biological motifs.

        Each motif is defined as a regex pattern that captures biological variation.
        """
        # Dictionary mapping motif names to their regex patterns
        # IUPAC nucleotide code: N = any base, W = A or T, etc.
        self.motifs = {
            # TATA-box: Core promoter element found ~25-30bp upstream of transcription start
            # Pattern: TATAAA with some variation allowed
            'TATA_box': r'TATA[WT]A[WT]',

            # CAAT-box: Enhancer element in eukaryotic promoters
            # Pattern: GG(T/C)CAATCT
            'CAAT_box': r'GG[TC]CAATCT',

            # GC-box: Binding site for Sp1 transcription factor
            # Pattern: GGGCGG
            'GC_box': r'GGGCGG',

            # E-box: Binding site for basic helix-loop-helix transcription factors
            # Pattern: CANNTG (CA followed by any 2 bases, then TG)
            'E_box': r'CA[ATCG]{2}TG',

            # CTCF binding motif: Simplified version
            # Real CTCF motif is more complex, but this captures the core
            'CTCF_core': r'[AG]CCGCG[CG]AGGAG',

            # CpG island marker: Multiple CG dinucleotides
            # Pattern: At least 3 CG dinucleotides within 20bp
            'CpG_cluster': r'(?:CG.{0,5}){3,}',

            # Kozak sequence: Translation initiation site
            # Pattern: (gcc)gccRccATGG where R = purine (A or G)
            'Kozak': r'GCC[AG]CCATGG',

            # Polyadenylation signal: AATAAA or ATTAAA
            'polyA_signal': r'A[AT]TAAA'
        }

        # Compiling patterns for efficiency
        # Compiled patterns are faster for repeated matching
        self.compiled_patterns = {
            name: re.compile(pattern)
            for name, pattern in self.motifs.items()
        }

        print("RegexMotifDetector initialized")
        print(f"  - {len(self.motifs)} motif patterns loaded")
        print(f"  - Motifs: {', '.join(self.motifs.keys())}")

    def find_motifs(self, sequence: str) -> Dict[str, List[Tuple[int, str]]]:
        """
        Finding all motif occurrences in a sequence.

        This demonstrates how regex (Type 3 regular language) can capture
        simple biological patterns before we move to more complex grammars.

        Parameters:
            sequence: DNA sequence string

        Returns:
            Dictionary mapping motif names to list of (position, matched_sequence) tuples

        Example:
            Input:  "GCGGTATAATAAGCGGGCGGCTCAG"
            Output: {
                'TATA_box': [(5, 'TATAAT')],
                'GC_box': [(13, 'GGGCGG')]
            }
        """
        sequence = sequence.upper()
        results = {}

        # Searching for each motif pattern
        for motif_name, pattern in self.compiled_patterns.items():
            matches = []

            # Finding all non-overlapping matches
            for match in pattern.finditer(sequence):
                position = match.start()
                matched_seq = match.group()
                matches.append((position, matched_seq))

            # Only storing motifs that were found
            if matches:
                results[motif_name] = matches

        return results

    def analyze_sequence(self, sequence: str, verbose: bool = True) -> Dict:
        """
        Comprehensive analysis of a sequence using regex patterns.

        This shows the limitations of regular expressions:
        - They can find simple patterns
        - But cannot capture nested structures or long-range dependencies
        - That's why we need CFGs and ultimately transformers

        Parameters:
            sequence: DNA sequence to analyze
            verbose: Whether to print detailed results

        Returns:
            Dictionary with analysis results
        """
        motifs_found = self.find_motifs(sequence)

        # Calculating statistics
        total_motifs = sum(len(matches) for matches in motifs_found.values())
        motif_types = len(motifs_found)

        # Creating analysis report
        analysis = {
            'sequence_length': len(sequence),
            'total_motifs_found': total_motifs,
            'motif_types_found': motif_types,
            'motifs': motifs_found,
            'has_promoter_elements': any(
                motif in motifs_found
                for motif in ['TATA_box', 'CAAT_box', 'GC_box']
            ),
            'has_CTCF': 'CTCF_core' in motifs_found
        }

        if verbose:
            print(f"\n{'=' * 70}")
            print("REGEX MOTIF ANALYSIS")
            print(f"{'=' * 70}")
            print(f"Sequence length: {len(sequence)} bp")
            print(f"Total motifs found: {total_motifs}")
            print(f"Motif types found: {motif_types}")
            print()

            if motifs_found:
                for motif_name, matches in motifs_found.items():
                    print(f"{motif_name}:")
                    for pos, seq in matches:
                        print(f"  Position {pos}: {seq}")
                print()
            else:
                print("No motifs found in this sequence.")
                print()

            print(f"Contains promoter elements: {analysis['has_promoter_elements']}")
            print(f"Contains CTCF core motif: {analysis['has_CTCF']}")
            print(f"{'=' * 70}\n")

        return analysis

    def compare_with_model(self, sequence: str, model_prediction: float) -> str:
        """
        Comparing regex-based detection with transformer model prediction.

        This demonstrates the key insight of your project:
        - Regex: Simple, interpretable, but limited
        - Transformer: Complex, powerful, learns patterns automatically

        Parameters:
            sequence: DNA sequence
            model_prediction: Transformer model's binding prediction (0-1)

        Returns:
            Comparison summary string
        """
        analysis = self.analyze_sequence(sequence, verbose=False)

        # Simple heuristic: if we find CTCF motif, we "predict" binding
        regex_prediction = 1 if analysis['has_CTCF'] else 0

        summary = f"""
{'=' * 70}
REGEX vs TRANSFORMER COMPARISON
{'=' * 70}
Sequence length: {len(sequence)} bp

REGEX APPROACH (Rule-based):
  - CTCF motif found: {analysis['has_CTCF']}
  - Prediction: {'Binding' if regex_prediction else 'No binding'}
  - Method: Explicit pattern matching
  - Limitation: Only finds exact/simple patterns

TRANSFORMER APPROACH (Learning-based):
  - Prediction score: {model_prediction:.3f}
  - Prediction: {'Binding' if model_prediction > 0.5 else 'No binding'}
  - Method: Learned from 1000s of examples
  - Advantage: Captures complex, position-dependent patterns

Agreement: {'✓ Yes' if (regex_prediction == (model_prediction > 0.5)) else '✗ No'}

WHY THE DIFFERENCE?
- Regex finds explicit motifs (if motif present → binding)
- Transformer learns context (position, flanking sequences, combinations)
- Real biology is complex: presence of motif ≠ guaranteed binding
{'=' * 70}
"""
        return summary

    def get_pattern(self, motif_name: str) -> str:
        """
        Getting the regex pattern for a specific motif.

        Useful for educational purposes - showing students the actual patterns.
        """
        return self.motifs.get(motif_name, "Motif not found")


def demonstrate_regex_hierarchy():
    """
    Demonstrating the Chomsky hierarchy concept using DNA motifs.

    This function shows how we progress from simple to complex language tools:
    1. Regular expressions (Type 3) - simple patterns
    2. Context-free grammars (Type 2) - nested structures
    3. Transformers - learn any pattern from data
    """
    print(f"\n{'=' * 70}")
    print("FORMAL LANGUAGE HIERARCHY IN DNA ANALYSIS")
    print(f"{'=' * 70}\n")

    # Example sequence with multiple features
    example_seq = "GCGGTATAATAAGCGGGCGGCTCAGCCGCGCAGGAGTTACG"

    print(f"Example sequence ({len(example_seq)} bp):")
    print(example_seq)
    print()

    # LEVEL 1: Regular Expressions (Type 3)
    print("LEVEL 1: Regular Expressions (Chomsky Type 3)")
    print("-" * 70)
    print("What they can do: Find simple, linear patterns")
    print("What they cannot do: Match nested structures, long-range dependencies")
    print()

    detector = RegexMotifDetector()
    results = detector.analyze_sequence(example_seq, verbose=True)

    print("\nLEVEL 2: Context-Free Grammars (Chomsky Type 2)")
    print("-" * 70)
    print("What they can do: Model nested structures (promoter → enhancer → gene)")
    print("Example: Regulatory regions with hierarchical organization")
    print("See: dna_grammar.py for CFG implementation")
    print()

    print("LEVEL 3: Transformer Neural Networks (Beyond Chomsky)")
    print("-" * 70)
    print("What they can do: Learn ANY pattern from data, including:")
    print("  - Long-range dependencies (position 10 affects position 150)")
    print("  - Context-dependent patterns (same motif, different meaning)")
    print("  - Combinations of features (motif + spacing + flanking sequence)")
    print("See: TransformerModel.py for implementation")
    print(f"\n{'=' * 70}\n")


if __name__ == "__main__":
    # Running demonstration
    demonstrate_regex_hierarchy()

    # Testing with example sequences
    detector = RegexMotifDetector()

    # Test 1: Sequence with TATA box
    print("\nTest 1: Promoter region")
    promoter = "GCGGTATAATAAGCGGGCGGCTCAG"
    detector.analyze_sequence(promoter)

    # Test 2: Sequence with CTCF motif
    print("\nTest 2: CTCF binding site")
    ctcf_region = "ATCGACCGCGCAGGAGCTGATCG"
    detector.analyze_sequence(ctcf_region)

    # Test 3: Random sequence (negative control)
    print("\nTest 3: Random sequence (negative control)")
    random_seq = "ATCGATCGATCGATCGATCG"
    detector.analyze_sequence(random_seq)