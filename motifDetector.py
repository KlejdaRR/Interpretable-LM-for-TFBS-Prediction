"""
Motif Detector using Regular Expressions

This module demonstrates the application of regular expressions to identify
known biological motifs in DNA sequences. These patterns represent well-studied
transcription factor binding sites discovered through decades of molecular biology.

Why Regular Expressions for DNA?
- DNA motifs can be represented as patterns over the alphabet {A, T, C, G}
- Regular expressions provide a formal way to define these patterns
- They allow for degeneracy (multiple allowed bases at a position)
"""

import re
from typing import List, Dict, Tuple


class MotifDetector:
    """
    A simple motif detector using regular expressions.

    This demonstrates formal language theory (regular languages) applied to genomics.
    Each motif pattern is a regular expression over the DNA alphabet.
    """

    def __init__(self):
        """
        Initialize with known biological motifs.

        Motif notation:
        - [AT] means "A or T at this position"
        - [ATCG] means "any nucleotide" (often written as N in biology)
        - Exact letters mean that specific base is required
        """

        # Known transcription factor binding motifs
        self.motifs = {
            # CTCF consensus sequence (the TF we're predicting!)
            # Pattern from JASPAR database
            'CTCF': r'CCGCGNGGNGGCAG',

            # TATA box - one of the most famous promoter elements
            # Found ~25-30bp upstream of transcription start
            'TATA_BOX': r'TATA[AT]A[AT]',

            # CAAT box - another common promoter element
            'CAAT_BOX': r'GGCCAATCT',

            # GC box - binding site for SP1 transcription factor
            'GC_BOX': r'GGGCGG',

            # E-box - binding site for bHLH transcription factors
            'E_BOX': r'CA[ATCG]{2}TG',

            # AP-1 binding site
            'AP1': r'TGA[GC]TCA'
        }

        # Compile regex patterns (N represents any nucleotide)
        self.compiled_patterns = {}
        for name, pattern in self.motifs.items():
            # Replace N with [ATCG] for regex
            regex_pattern = pattern.replace('N', '[ATCG]')
            self.compiled_patterns[name] = re.compile(regex_pattern)

        print("Motif Detector initialized with patterns:")
        for name, pattern in self.motifs.items():
            print(f"  {name}: {pattern}")

    def find_motifs(self, sequence: str) -> Dict[str, List[Tuple[int, str]]]:
        """
        Find all known motifs in a DNA sequence.

        Args:
            sequence: DNA sequence string (e.g., "ATCGATCG...")

        Returns:
            Dictionary mapping motif names to list of (position, matched_sequence) tuples

        Example:
            Input: "ATCGTATAAATGCTAGC"
            Output: {'TATA_BOX': [(5, 'TATAAAT')]}
        """
        results = {}

        for motif_name, pattern in self.compiled_patterns.items():
            matches = []

            # Find all matches in the sequence
            for match in pattern.finditer(sequence):
                position = match.start()
                matched_seq = match.group()
                matches.append((position, matched_seq))

            if matches:
                results[motif_name] = matches

        return results

    def has_motif(self, sequence: str, motif_name: str) -> bool:
        """
        Check if a specific motif exists in the sequence.

        Args:
            sequence: DNA sequence
            motif_name: Name of motif to search for

        Returns:
            True if motif found, False otherwise
        """
        if motif_name not in self.compiled_patterns:
            return False

        return self.compiled_patterns[motif_name].search(sequence) is not None

    def get_motif_features(self, sequence: str) -> Dict[str, int]:
        """
        Extract motif presence as binary features.

        This can be used as additional features for the ML model.

        Args:
            sequence: DNA sequence

        Returns:
            Dictionary mapping motif names to binary values (1=present, 0=absent)
        """
        features = {}

        for motif_name in self.motifs.keys():
            features[f'has_{motif_name}'] = int(self.has_motif(sequence, motif_name))

        return features

    def analyze_sequence(self, sequence: str, name: str = "Sequence") -> None:
        """
        Print a detailed analysis of motifs in a sequence.

        Args:
            sequence: DNA sequence to analyze
            name: Optional name for the sequence
        """
        print(f"\n{'=' * 70}")
        print(f"Motif Analysis: {name}")
        print(f"{'=' * 70}")
        print(f"Sequence length: {len(sequence)} bp")
        print(f"Sequence: {sequence[:60]}{'...' if len(sequence) > 60 else ''}")

        motifs_found = self.find_motifs(sequence)

        if motifs_found:
            print(f"\nMotifs found: {len(motifs_found)}")
            for motif_name, matches in motifs_found.items():
                print(f"\n  {motif_name}:")
                for position, matched_seq in matches:
                    print(f"    Position {position}: {matched_seq}")
        else:
            print("\nNo known motifs detected")

        print(f"{'=' * 70}")


def demonstrate_regex_motifs():
    """
    Demonstration function showing how regex patterns work on DNA.

    This illustrates the formal language theory concept:
    Regular expressions define regular languages over the DNA alphabet.
    """
    print("\n" + "=" * 70)
    print("DEMONSTRATION: Regular Expressions for DNA Motifs")
    print("=" * 70)

    detector = MotifDetector()

    # Example sequences
    test_sequences = {
        "CTCF binding site": "ATCGCCGCGTAGGAGGCAGTGCTAGC",
        "TATA box promoter": "GCTAGCTATAAAGGCTAGCTA",
        "Multiple motifs": "GGGCGGATATATAAGGCCAATCT",
        "No known motifs": "ATATATATATATATATATAT"
    }

    for name, seq in test_sequences.items():
        detector.analyze_sequence(seq, name)


if __name__ == "__main__":
    demonstrate_regex_motifs()