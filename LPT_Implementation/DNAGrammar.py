"""
DNA Context-Free Grammar

This module implements a context-free grammar (CFG) for modeling DNA regulatory regions.
This demonstrates the next level in the Chomsky hierarchy after regular expressions.

What is a Context-Free Grammar?
A CFG is a set of recursive rules that can generate and parse nested structures.
Example: S → aSb | ε generates strings like "ab", "aabb", "aaabbb"

Why CFG for DNA?
Regulatory regions have hierarchical structure:
- Promoter contains: TATA-box + CAAT-box + Transcription Start Site
- Enhancer contains: Multiple transcription factor binding sites
- Gene contains: Promoter + Exons + Introns

Regular expressions cannot capture this nested, hierarchical organization.

Connection to Language Processing Technologies course:
- Context-free grammars and pushdown automata
- Parsing algorithms
- Chomsky hierarchy (Type 2 languages)
- Lark parser generator (similar to ANTLR, Yacc)
"""

from lark import Lark, Transformer, Tree
from typing import Dict, List, Optional


class DNAGrammar:
    """
    A context-free grammar for DNA regulatory regions.

    This demonstrates how CFGs can model the hierarchical structure of
    genomic elements - something regex cannot do.
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
            // This shows hierarchical composition: promoter CONTAINS elements
            promoter: "TATA" elements "TSS"

            // Elements: can have multiple components
            // This demonstrates nested structure
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
        # Lark will build a pushdown automaton from our grammar
        self.parser = Lark(
            self.grammar_definition,
            start='regulatory_region',  # Starting symbol
            parser='lalr',  # LALR parsing algorithm (efficient)
            keep_all_tokens=True  # Keep all tokens for analysis
        )

        print("DNA Context-Free Grammar initialized")
        print("  - Grammar type: Type 2 (Context-Free)")
        print("  - Parser algorithm: LALR")
        print("  - Structures: Promoter, Enhancer, TFBS")

    def parse(self, sequence_structure: str) -> Optional[Tree]:
        """
        Parsing a DNA structure description into a parse tree.

        This demonstrates how CFGs can recognize hierarchical structure.

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

        This is what CFGs excel at: recognizing valid structures.

        Parameters:
            sequence_structure: Structure to validate

        Returns:
            True if valid according to grammar, False otherwise
        """
        tree = self.parse(sequence_structure)
        return tree is not None

    def analyze_structure(self, sequence_structure: str, verbose: bool = True) -> Dict:
        """
        Analyzing the hierarchical structure of a regulatory region.

        This demonstrates the power of CFGs over regex:
        - CFGs can recognize nested structures
        - CFGs can enforce ordering rules
        - CFGs can model composition (promoter contains elements)

        Parameters:
            sequence_structure: Structure string to analyze
            verbose: Whether to print detailed analysis

        Returns:
            Dictionary with structural analysis
        """
        tree = self.parse(sequence_structure)

        if tree is None:
            return {
                'valid': False,
                'error': 'Failed to parse - structure does not match grammar'
            }

        # Analyzing the parse tree
        analysis = {
            'valid': True,
            'type': None,
            'components': [],
            'has_promoter': False,
            'has_enhancer': False,
            'promoter_elements': [],
            'tfbs_count': 0
        }

        # Walking the tree to extract information
        for child in tree.children:
            if hasattr(child, 'data'):
                if child.data == 'promoter':
                    analysis['has_promoter'] = True
                    analysis['type'] = 'promoter' if not analysis['has_enhancer'] else 'promoter+enhancer'

                    # Extracting promoter elements
                    for element in child.children:
                        if hasattr(element, 'data'):
                            elem_type = element.data
                            if elem_type in ['tata_box', 'caat_box', 'gc_box', 'tss']:
                                analysis['promoter_elements'].append(elem_type)

                elif child.data == 'enhancer':
                    analysis['has_enhancer'] = True
                    analysis['type'] = 'enhancer' if not analysis['has_promoter'] else 'promoter+enhancer'

                    # Counting TFBS in enhancer
                    analysis['tfbs_count'] = self._count_tfbs(child)

        if verbose:
            self._print_analysis(analysis, tree)

        return analysis

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


class RegionClassifier(Transformer):
    """
    A Lark Transformer that classifies regulatory regions.

    This demonstrates how CFG parse trees can be transformed into
    structured data for downstream analysis.
    """

    def regulatory_region(self, items):
        """Transforming regulatory_region rule"""
        return {
            'type': 'regulatory_region',
            'components': items
        }

    def promoter(self, items):
        """Transforming promoter rule"""
        elements = [item for item in items if item is not None]
        return {
            'type': 'promoter',
            'elements': elements
        }

    def enhancer(self, items):
        """Transforming enhancer rule"""
        return {
            'type': 'enhancer',
            'tfbs_sites': items
        }


