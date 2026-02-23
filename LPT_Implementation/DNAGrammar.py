"""
DNA Context-Free Grammar

This module implements a context-free grammar (CFG) for modeling DNA regulatory regions.
CFGs represent the next level in the Chomsky hierarchy after regular expressions.

What is a Context-Free Grammar?
A CFG is a set of recursive rules that can generate and parse nested structures.

Why are we using CFG for DNA?
Regulatory regions have hierarchical structure:
- Promoter contains: TATA-box + CAAT-box + Transcription Start Site
- Enhancer contains: Multiple transcription factor binding sites
- Gene contains: Promoter + Exons + Introns

Therefore, regular expressions cannot capture this nested, hierarchical organization.
"""

from lark import Lark, Tree
from typing import Dict, Optional


class DNAGrammar:
    """
    This class demonstrates how CFGs can model the hierarchical structure of
    genomic elements.
    """

    def __init__(self):
        """
        Initializing the DNA grammar using Lark parser generator.

        The grammar defines the structure of regulatory regions:
        - regulatory_region: The top-level structure
        - promoter: Contains core promoter elements
        - enhancer: Contains TF binding sites
        - motif: Individual binding sites
        - sequence: Raw DNA sequence
        """

        self.grammar_definition = r"""
            // A regulatory region is either a promoter or an enhancer
            
            regulatory_region: promoter | enhancer

            // Promoter: TATA-box, optional elements, then TSS
            // This shows hierarchical composition because promoter CONTAINS elements
            promoter: "TATA" elements "TSS"

            // Elements: can have multiple components
            elements: element*

            element: "CAAT" | "GC" | "SPACER"

            // Enhancer: Multiple TFBS (at least 2)
            // Different structure from promoter
            enhancer: tfbs tfbs tfbs*

            tfbs: "CTCF" | "EBOX"

            %import common.WS
            %ignore WS
        """

        # Creating the parser
        self.parser = Lark(
            self.grammar_definition,
            start='regulatory_region',  # Starting symbol
            parser='lalr',  # LALR parsing algorithm (efficient)
            keep_all_tokens=True  # Keep all tokens for analysis
        )

        print("DNA Context-Free Grammar initialized")
        print("  - Parser algorithm: LALR")
        print("  - Structures: Promoter, Enhancer, TFBS")

    def parse(self, sequence_structure: str) -> Optional[Tree]:
        """
        Method for parsing a DNA structure description into a parse tree.

        Parameters:
            sequence_structure: String describing regulatory region structure
                              Example: "TATAWAW SPACER GGYCAATCT SPACER TSS"
        Returns:
            Parse tree if valid, None if parsing fails

        Example:
            Input:  "TATAWAW ATCG GGYCAATCT GCGC TSS"
            Output: Tree showing hierarchical structure:
                    regulatory_region
                    └── promoter
                        ├── tata_box
                        ├── spacer
                        ├── caat_box
                        ├── spacer
                        └── tss
        """
        try:
            tree = self.parser.parse(sequence_structure)
            return tree
        except Exception as e:
            print(f"Parsing failed: {e}")
            return None

    def validate_structure(self, sequence_structure: str) -> bool:
        """
        Validating whether a sequence matches the regulatory region grammar.

        Parameters:
            sequence_structure: Structure to validate

        Returns:
            True if valid according to grammar, False otherwise
        """
        tree = self.parse(sequence_structure)
        return tree is not None

    def _count_tfbs(self, enhancer_node) -> int:
        """Counting transcription factor binding sites in enhancer"""
        count = 0
        for child in enhancer_node.iter_subtrees():
            if hasattr(child, 'data') and child.data == 'tfbs':
                count += 1
        return count

    def _print_analysis(self, analysis: Dict, tree: Tree):
        """Printing detailed structural analysis"""
        print(f"\n{'=' * 70}")
        print("CFG STRUCTURAL ANALYSIS")
        print(f"{'=' * 70}")
        print(f"Valid structure: {analysis['valid']}")
        print(f"Type: {analysis['type']}")
        print()

        if analysis['has_promoter']:
            print("PROMOTER DETECTED:")
            print(f"  Elements found: {', '.join(analysis['promoter_elements'])}")
            print()

        if analysis['has_enhancer']:
            print("ENHANCER DETECTED:")
            print(f"  Number of TFBS: {analysis['tfbs_count']}")
            print()

        print("PARSE TREE:")
        print(tree.pretty())
        print(f"{'=' * 70}\n")



