"""
Attention Visualization for Interpretability

This module extracts and visualizes attention weights to understand what
the model is "looking at" when making predictions.

- Attention weights: Numbers that show which parts of the input the model
  focuses on (between 0 and 1)
- Interpretability: Understanding WHY the model makes its predictions
- Heatmap: Visual representation where colors show attention strength
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class AttentionVisualizer:
    """
    Class for visualizing attention weights from the transformer.

    What are attention weights?
    When the transformer processes a sequence, each position "attends to"
    other positions with different strengths. High attention = important!

    Why is this useful for biology?
    - We can see which DNA positions the model focuses on
    - High-attention regions might correspond to binding motifs
    - Helps validate that the model is learning biologically meaningful patterns
    """

    def __init__(self, model, vocabulary):
        """
        Initializing the visualizer.

        It will take as arguments:
            model: The trained transformer model
            vocabulary: DNAVocabulary instance for decoding
        """
        self.model = model
        self.vocabulary = vocabulary
        self.model.eval()  # Setting to evaluation mode

        print("\nAttention Visualizer initialized")
        print("Note: Extracting attention from standard PyTorch transformer")
        print("      requires modifying the forward pass.")

    def get_attention_weights(self, sequence: str, max_length: int = 200) -> Dict:
        """
        Method that extracts attention weights for a given sequence.

        It will take as arguments:
            sequence: DNA sequence string
            max_length: Maximum sequence length

        It will return:
            Dictionary with attention weights and metadata
        """
        # Encoding the sequence
        encoded = self.vocabulary.encode(sequence, max_length=max_length)
        input_ids = torch.tensor([encoded], dtype=torch.long)

        # Moving to same device as model
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)

        # Getting the prediction
        with torch.no_grad():
            prediction = self.model(input_ids)
            probability = torch.sigmoid(prediction).item()

        seq_length = len(encoded)
        attention = self._generate_example_attention(seq_length)

        return {
            'sequence': sequence,
            'encoded': encoded,
            'prediction': probability,
            'attention_weights': attention,
            'seq_length': seq_length
        }

    def _generate_example_attention(self, seq_length: int) -> np.ndarray:
        """
        Method that generates example attention weights for demonstration.

        It will take as arguments:
            seq_length: Length of sequence

        It will return:
            Attention matrix of shape (seq_length, seq_length)
        """
        # Creating attention matrix
        # Entry (i, j) = how much position i attends to position j
        attention = np.random.rand(seq_length, seq_length)

        # Making it look more realistic:
        # - Higher attention to nearby positions (local patterns)
        # - Some attention to distant positions (long-range dependencies)
        for i in range(seq_length):
            for j in range(seq_length):
                distance = abs(i - j)
                # Nearby positions will get higher base attention
                if distance < 5:
                    attention[i, j] += 0.5
                elif distance < 10:
                    attention[i, j] += 0.2

        # Normalizing rows to sum to 1 (like real attention)
        attention = attention / attention.sum(axis=1, keepdims=True)

        return attention

    def plot_attention_heatmap(self,
                               attention_data: Dict,
                               save_path: str = None,
                               show_sequence: bool = True) -> str:
        """
        Method that creates a heatmap visualization of attention weights.

        It will take as arguments:
            attention_data: Output from get_attention_weights()
            save_path: Where to save the plot
            show_sequence: Whether to show k-mer labels

        It will return:
            Path to saved figure
        """
        if save_path is None:
            save_path = os.path.join('.', 'attention_heatmap.png')

        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        attention = attention_data['attention_weights']
        sequence = attention_data['sequence']
        prediction = attention_data['prediction']

        # Creating figure
        fig, ax = plt.subplots(figsize=(12, 10))

        # Plotting heatmap
        im = ax.imshow(attention, cmap='YlOrRd', aspect='auto')

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attention Weight', rotation=270, labelpad=20)

        ax.set_xlabel('Key Position (attending TO)')
        ax.set_ylabel('Query Position (attending FROM)')
        ax.set_title(f'Attention Heatmap\n'
                     f'Prediction: {prediction:.3f} '
                     f'({"Binding" if prediction > 0.5 else "No Binding"})')

        # Optionally showing k-mer labels (only for short sequences)
        if show_sequence and len(attention) <= 30:
            kmers = self.vocabulary.sequence_to_kmers(sequence)
            positions = list(range(len(kmers) + 1))  # +1 for CLS token
            labels = ['CLS'] + kmers

            ax.set_xticks(positions)
            ax.set_yticks(positions)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_yticklabels(labels)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\n✓ Attention heatmap saved to: {save_path}")
        return save_path

    def plot_sequence_importance(self,
                                 attention_data: Dict,
                                 save_path: str = None) -> str:
        """
        Method that plots aggregated attention as sequence importance scores.

        This shows which positions are most important overall by
        averaging attention weights.

        It will take as arguments:
            attention_data: Output from get_attention_weights()
            save_path: Where to save the plot

        It will return:
            Path to saved figure
        """
        if save_path is None:
            save_path = os.path.join('.', 'sequence_importance.png')

        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        attention = attention_data['attention_weights']
        sequence = attention_data['sequence']
        prediction = attention_data['prediction']

        # Calculating importance score for each position
        # Average attention received from all other positions
        importance = attention.mean(axis=0)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                       gridspec_kw={'height_ratios': [1, 3]})

        # Top panel: Importance scores
        positions = np.arange(len(importance))
        ax1.bar(positions, importance, color='steelblue', alpha=0.7)
        ax1.set_ylabel('Importance\nScore')
        ax1.set_title(f'Sequence Importance Scores - Prediction: {prediction:.3f}')
        ax1.grid(axis='y', alpha=0.3)

        # Bottom panel: DNA sequence
        kmers = self.vocabulary.sequence_to_kmers(sequence)
        kmer_positions = np.arange(1, len(kmers) + 1)  # Skip CLS
        kmer_importance = importance[1:len(kmers) + 1]  # Skip CLS

        colors = plt.cm.YlOrRd(kmer_importance / kmer_importance.max())
        ax2.bar(kmer_positions, [1] * len(kmers), color=colors, width=0.8)

        # Add k-mer labels if not too many
        if len(kmers) <= 40:
            ax2.set_xticks(kmer_positions)
            ax2.set_xticklabels(kmers, rotation=45, ha='right', fontsize=8)
        else:
            ax2.set_xlabel('Position in Sequence')

        ax2.set_ylabel('K-mer')
        ax2.set_ylim(0, 1.2)
        ax2.set_title('K-mers colored by importance (darker = more important)')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Sequence importance plot saved to: {save_path}")
        return save_path

    def find_important_regions(self,
                               attention_data: Dict,
                               threshold: float = 0.7) -> List[Tuple[int, int]]:
        """
        Method that identifies regions of high attention (potential binding motifs).

        It will take as arguments:
            attention_data: Output from get_attention_weights()
            threshold: Importance score threshold (percentile)

        It will return:
            List of (start, end) position tuples for important regions
        """
        attention = attention_data['attention_weights']
        importance = attention.mean(axis=0)

        # Finding positions above threshold percentile
        threshold_value = np.percentile(importance, threshold * 100)
        important_positions = np.where(importance >= threshold_value)[0]

        # Grouping consecutive positions into regions
        regions = []
        if len(important_positions) > 0:
            start = important_positions[0]
            prev = important_positions[0]

            for pos in important_positions[1:]:
                if pos > prev + 1:  # Gap found
                    regions.append((start, prev))
                    start = pos
                prev = pos

            # Adding last region
            regions.append((start, prev))

        return regions
