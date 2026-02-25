"""
Regular Expression Motif Detector

This class demonstrates how regular expressions can be used to detect known
biological motifs in DNA sequences, the first level of language hierarchy.

What is a motif?
A motif is a recurring pattern in DNA that has biological significance.
Example: The TATA-box is a DNA sequence that helps position RNA polymerase.
"""

import re
from typing import Dict


class RegexMotifDetector:
    """Class that detects biological motifs using regular expressions.
    """

    def __init__(self):
        """
        Initializing the detector with common biological motifs.

        Each motif is defined as a regex pattern that captures biological variation.
        """
        # Dictionary mapping motif names to their regex patterns
        # W = A or T (Weak bonds)
        # R = A or G (purine)
        # Y = C or T (pyrimidine)
        # N = Any base(A, C, G, T)

        self.motifs = {
            # TATA-box: Core promoter element found upstream of transcription start
            # Pattern: Positions 1-4: Exactly TATA
            #          Position 5: Either W (A or T) or T (so: A or T)
            #          Position 6: Exactly A
            #          Position 7: Either W (A or T) or T (so: A or T)
            'TATA_box': r'TATA[WT]A[WT]',

            # CAAT-box: Enhancer element in eukaryotic promoters
            # Pattern: Positions 1-2: Exactly GG
            #          Position 3: Either T or C
            #          Positions 4-9: Exactly CAATCT
            'CAAT_box': r'GG[TC]CAATCT',

            # GC-box: Binding site for Sp1 transcription factor
            # Pattern: Exactly GGGCGG
            'GC_box': r'GGGCGG',

            # E-box: Binding site for basic helix-loop-helix transcription factors
            # Pattern: Positions 1-2: Exactly CA
            #          Position 3: Any 2 bases from A,T,C,G
            #          Positions 4-5: Exactly TG
            'E_box': r'CA[ATCG]{2}TG',

            # CTCF binding motif: Simplified version
            # Patter: Position 1: Either A or G
            #         Positions 2-6: Exactly CCGCG
            #         Position 7: Either C or G
            #         Positions 8-12: Exactly AGGAG
            'CTCF_core': r'[AG]CCGCG[CG]AGGAG',

            # Kozak sequence: Translation initiation site
            # Pattern: Positions 1-3: Exactly GCC
            #          Position 4: Either A or G
            #          Positions 5-10: Exactly CCATGG
            'Kozak': r'GCC[AG]CCATGG',

            # Polyadenylation signal: AATAAA or ATTAAA
            #Pattern: Position 1: Exactly A
            #         Position 2: Either A or T
            #         Positions 3-6: Exactly TAAA
            'polyA_signal': r'A[AT]TAAA'
        }

        # using re.compile() method in order to store the compiled patterns in the class
        # instance in order to reuse them everywhere
        self.compiled_patterns = {
            name: re.compile(pattern)
            for name, pattern in self.motifs.items()
        }

        print("RegexMotifDetector initialized")
        print(f"  - Motifs: {', '.join(self.motifs.keys())}")

    def find_motifs(self, sequence):
        """
        Finding all motif occurrences in a sequence.

        Parameters:
            sequence: DNA sequence string

        Returns:
            Dictionary mapping motif names to list of (position, matched_sequence) tuples
        """
        sequence = sequence.upper()
        results = {}

        for motif_name, pattern in self.compiled_patterns.items():
            matches = []

            # finditer() gives a match iterator
            # Each match is an object with .start() and .group() methods
            for match in pattern.finditer(sequence):
                # match.start() = where in the sequence the match begins
                # match.group() = the actual DNA sequence that matched
                position = match.start()
                matched_sequence = match.group()

                # appending to store each of the groups of DNA sequences where the motif was found
                matches.append((position, matched_sequence))

            if matches:
                results[motif_name] = matches

        return results

    def analyze_sequence(self, sequence: str, verbose: bool = True) -> Dict:
        """
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

