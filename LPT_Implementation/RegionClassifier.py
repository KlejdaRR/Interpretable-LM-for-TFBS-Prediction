from lark import Transformer

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