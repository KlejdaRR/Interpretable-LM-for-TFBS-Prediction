"""
DNA Regulatory Region Grammar using Context-Free Grammar (CFG)

This module demonstrates how Context-Free Grammars can model the hierarchical
structure of DNA regulatory regions. Unlike regular expressions (which can only
capture linear patterns), CFGs can represent nested and hierarchical structures.

Why CFG for DNA?
- Regulatory regions have hierarchical structure (promoter contains TATA box, etc.)
- CFGs are more expressive than regular languages
- They can model the compositional nature of regulatory elements

We use the Lark parser generator to implement and parse the grammar.
"""

from lark import Lark, Tree, Token
from typing import Dict, List, Optional

# Define the grammar for DNA regulatory regions
DNA_GRAMMAR = r"""
    // Start rule: A regulatory region can be a promoter or an enhancer
    regulatory_region: promoter | enhancer

    // Promoter structure: core promoter elements in order
    // Typically: TATA box + Initiator + transcription start site
    promoter: core_promoter_element+ transcription_start

    // Core promoter can contain known motifs
    core_promoter_element: tata_box | caat_box | gc_box

    // Enhancer: collection of transcription factor binding sites
    enhancer: tf_binding_site+

    // Individual motif patterns (simplified for grammar)
    tata_box: "TATA" wildcard_base wildcard_base
    caat_box: "GGCCAATCT"
    gc_box: "GGGCGG"

    // Generic TF binding site
    tf_binding_site: strong_binding | weak_binding
    strong_binding: conserved_core flanking_region
    weak_binding: degenerate_sequence

    // Building blocks
    conserved_core: base base base base base
    flanking_region: base base base
    degenerate_sequence: base base base base
    transcription_start: "ATG"  // Start codon approximation

    // DNA bases (terminal symbols)
    base: "A" | "T" | "C" | "G"
    wildcard_base: "A" | "T" | "C" | "G"

    // Whitespace handling
    %import common.WS
    %ignore WS
"""


class DNAGrammarParser:
    """
    A parser for DNA regulatory regions using Context-Free Grammar.

    This demonstrates formal language theory concepts:
    - Terminal symbols: {A, T, C, G}
    - Non-terminal symbols: promoter, enhancer, tata_box, etc.
    - Production rules: how complex structures are built from simpler ones
    """

    def __init__(self):
        """Initialize the Lark parser with our DNA grammar."""
        self.parser = Lark(DNA_GRAMMAR, start='regulatory_region')
        print("DNA Grammar Parser initialized")
        print("Grammar supports: promoters, enhancers, and TF binding sites")

    def parse(self, sequence: str) -> Optional[Tree]:
        """
        Parse a DNA sequence according to the grammar.

        Args:
            sequence: DNA sequence string with spaces between elements
                     (spaces help delimit grammar elements)

        Returns:
            Parse tree if successful, None if parsing fails

        Note: The sequence needs to match our grammar structure.
              This is a simplified demonstration - real DNA is more complex!
        """
        try:
            tree = self.parser.parse(sequence)
            return tree
        except Exception as e:
            return None

    def identify_structure(self, sequence: str) -> Dict[str, any]:
        """
        Identify the regulatory structure of a DNA sequence.

        This is a simplified version that looks for grammar-defined patterns.

        Args:
            sequence: DNA sequence

        Returns:
            Dictionary with structural information
        """
        result = {
            'has_tata_box': False,
            'has_caat_box': False,
            'has_gc_box': False,
            'has_start_codon': False,
            'structure_type': 'unknown'
        }

        # Check for known motifs (this is a simplification)
        if 'TATA' in sequence:
            result['has_tata_box'] = True
        if 'GGCCAATCT' in sequence:
            result['has_caat_box'] = True
        if 'GGGCGG' in sequence:
            result['has_gc_box'] = True
        if 'ATG' in sequence:
            result['has_start_codon'] = True

        # Determine structure type based on elements
        if result['has_tata_box'] or result['has_caat_box']:
            result['structure_type'] = 'promoter-like'
        elif result['has_gc_box']:
            result['structure_type'] = 'enhancer-like'

        return result

    def describe_grammar(self) -> None:
        """Print a description of the grammar structure."""
        print("\n" + "=" * 70)
        print("DNA REGULATORY REGION GRAMMAR")
        print("=" * 70)
        print("\nGrammar Hierarchy:")
        print("  regulatory_region")
        print("  ├── promoter")
        print("  │   ├── core_promoter_element (TATA, CAAT, GC boxes)")
        print("  │   └── transcription_start (ATG)")
        print("  └── enhancer")
        print("      └── tf_binding_site (conserved or degenerate)")
        print("\nTerminal Alphabet: Σ = {A, T, C, G}")
        print("=" * 70)


class SimpleRegulatoryAnalyzer:
    """
    A simplified analyzer that identifies regulatory elements without strict parsing.

    This provides a practical middle ground between regex and full CFG parsing,
    demonstrating how hierarchical structure can be recognized.
    """

    def __init__(self):
        """Initialize the analyzer with known motif patterns."""
        self.motifs = {
            'TATA_box': ['TATAAA', 'TATAAT', 'TATAAG', 'TATATA'],
            'CAAT_box': ['GGCCAATCT'],
            'GC_box': ['GGGCGG'],
            'Inr': ['YYANWYY'],  # Initiator element (Y=pyrimidine, N=any)
            'START_codon': ['ATG']
        }

    def analyze_regulatory_region(self, sequence: str) -> Dict[str, any]:
        """
        Analyze a sequence for regulatory structure.

        This demonstrates hierarchical composition:
        - Core elements (motifs)
        - Composite structures (promoter = TATA + Inr + START)
        - Region classification (promoter vs enhancer)

        Args:
            sequence: DNA sequence to analyze

        Returns:
            Dictionary with hierarchical structure information
        """

        # Level 1: Identify core elements (motifs)
        elements = {
            'TATA_box': any(motif in sequence for motif in self.motifs['TATA_box']),
            'CAAT_box': any(motif in sequence for motif in self.motifs['CAAT_box']),
            'GC_box': any(motif in sequence for motif in self.motifs['GC_box']),
            'START_codon': 'ATG' in sequence
        }

        # Level 2: Identify composite structures
        is_promoter = (elements['TATA_box'] or elements['CAAT_box']) and elements['START_codon']
        is_enhancer = elements['GC_box'] or (elements['TATA_box'] and not elements['START_codon'])

        # Level 3: Overall classification
        if is_promoter:
            region_type = 'PROMOTER'
            description = "Contains core promoter elements with transcription start"
        elif is_enhancer:
            region_type = 'ENHANCER'
            description = "Contains enhancer elements (distal regulatory region)"
        else:
            region_type = 'UNCLASSIFIED'
            description = "No clear regulatory structure identified"

        return {
            'region_type': region_type,
            'description': description,
            'core_elements': elements,
            'hierarchical_structure': {
                'level_1_elements': [k for k, v in elements.items() if v],
                'level_2_composition': {
                    'is_promoter': is_promoter,
                    'is_enhancer': is_enhancer
                }
            }
        }

    def print_analysis(self, sequence: str, name: str = "Sequence") -> None:
        """Print detailed regulatory analysis."""
        print(f"\n{'=' * 70}")
        print(f"Regulatory Region Analysis: {name}")
        print(f"{'=' * 70}")
        print(f"Sequence: {sequence[:60]}{'...' if len(sequence) > 60 else ''}")

        analysis = self.analyze_regulatory_region(sequence)

        print(f"\nRegion Type: {analysis['region_type']}")
        print(f"Description: {analysis['description']}")

        print("\nCore Elements Detected:")
        for element, present in analysis['core_elements'].items():
            status = "✓" if present else "✗"
            print(f"  {status} {element}")

        print("\nHierarchical Structure:")
        struct = analysis['hierarchical_structure']
        print(
            f"  Level 1 (Elements): {', '.join(struct['level_1_elements']) if struct['level_1_elements'] else 'None'}")
        print(f"  Level 2 (Composition):")
        print(f"    - Is Promoter: {struct['level_2_composition']['is_promoter']}")
        print(f"    - Is Enhancer: {struct['level_2_composition']['is_enhancer']}")

        print(f"{'=' * 70}")


def demonstrate_cfg():
    """
    Demonstration of Context-Free Grammar for DNA.

    Shows how CFG can model hierarchical structure beyond what regex can capture.
    """
    print("\n" + "=" * 70)
    print("DEMONSTRATION: Context-Free Grammar for DNA Regulatory Regions")
    print("=" * 70)

    # Initialize grammar parser
    grammar_parser = DNAGrammarParser()
    grammar_parser.describe_grammar()

    # Initialize practical analyzer
    print("\n" + "-" * 70)
    print("Practical Hierarchical Analysis")
    print("-" * 70)

    analyzer = SimpleRegulatoryAnalyzer()

    # Test sequences
    test_sequences = {
        "Promoter example": "GCTAGCTATAAAGGCTAGCTATGCGATCG",  # Has TATA and ATG
        "Enhancer example": "ATCGGGGCGGTAGCTAGCTAG",  # Has GC box
        "Core promoter": "GGCCAATCTATGATCGATCG",  # Has CAAT and ATG
        "No structure": "ATATATATATCGCGCGCG"  # Random
    }

    for name, seq in test_sequences.items():
        analyzer.print_analysis(seq, name)


if __name__ == "__main__":
    demonstrate_cfg()