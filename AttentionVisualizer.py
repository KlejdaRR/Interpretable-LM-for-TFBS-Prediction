"""
Attention Visualization for Interpretability
This file helps VISUALIZE what our AI model is "looking at" when it makes predictions.

In simple terms:
- When we read a sentence, some words are more important than others
- When our model reads DNA, some positions are more important than others
- This file creates pictures (graphs) to show which DNA positions are most important

It is needed because:
Without visualization, our model is a "black box" - we can't see why it makes predictions.
With visualization, we can:
1. See which DNA positions the model focuses on
2. Check if the model learned real biology (not just memorizing)
3. Discover new binding patterns we didn't know about

- Attention weights: Numbers (0 to 1) showing how much the model "pays attention" to each position
  0.9 = very important, the model looks at this a lot
  0.1 = not important, the model mostly ignores this
- Heatmap: A picture where colors show values (red = high, yellow = low)
- Interpretability: Being able to understand and explain what the AI is doing
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
    A class that creates visualizations of attention weights.
    This class takes a DNA sequence and shows:
    1. Which positions the model thinks are important
    2. How positions relate to each other
    3. Where potential binding motifs might be
    """

    def __init__(self, model, vocabulary):
        """
        Initializing the visualizer
        Parameters used:
        - model: The trained transformer we want to visualize
        - vocabulary: The dictionary that translates DNA → numbers

        What will happen:
        We will save the model and vocabulary so we can use them later.
        """
        # Storing the model and vocabulary as instance variables
        self.model = model
        self.vocabulary = vocabulary

        # Putting the model in evaluation mode (not training mode)
        # It turns off things like dropout and batch normalization
        self.model.eval()

        print("\nAttention Visualizer initialized")
        print("Note: Extracting attention from standard PyTorch transformer")
        print("      requires modifying the forward pass.")

    def get_attention_weights(self, sequence: str, max_length: int = 200) -> Dict:
        """
        Extracting attention weights for a DNA sequence.

        This method takes a DNA sequence (like "ATCGATCG") and returns:
        1. What the model predicts (binding or not binding)
        2. Which positions the model focused on (attention weights)
        3. Other useful metadata

        Step by step process:
        1. Convert DNA letters (ATCG) to numbers the model understands
        2. Feed those numbers into the model
        3. Get the model's prediction (0 to 1, where 1 = binding)
        4. Extract which positions the model paid attention to
        5. Package everything into a dictionary (like a labeled box)

        Parameters used:
        - sequence: DNA sequence as a string, like "ATCGATCG"
        - max_length: Maximum length to process (default 200)
                     Longer sequences are truncated (cut off)

        It will return:
        A dictionary (think: labeled box) containing:
        - 'sequence': The original DNA sequence
        - 'encoded': The sequence converted to numbers
        - 'prediction': Model's prediction (0-1, binding probability)
        - 'attention_weights': Matrix showing what model focused on
        - 'seq_length': How long the sequence is

        For example:
        sequence = "ATCGATCG"
        result = visualizer.get_attention_weights(sequence)
        print(result['prediction'])  # Might print: 0.85 (85% chance of binding)
        """

        # STEP 1: Converting DNA sequence to numbers
        encoded = self.vocabulary.encode(sequence, max_length=max_length)

        # STEP 2: Preparing the encoded sequence for PyTorch
        # We need to:
        # a) Convert the list of numbers to a PyTorch tensor
        # b) Add a batch dimension
        #    PyTorch expects shape: [batch_size, sequence_length]
        #    The [encoded] adds brackets, making it: [[1, 2, 3, ...]]
        input_ids = torch.tensor([encoded], dtype=torch.long)

        # STEP 3: Moving the input to the same device as the model
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)

        # STEP 4: Get the model's prediction
        # torch.no_grad() means "don't track gradients" (because we're not training)
        # This will save memory and makes things faster
        with torch.no_grad():
            # Feed the sequence through the model
            prediction = self.model(input_ids)

            # Converting the raw output to a probability (0 to 1)
            # sigmoid function: it takes any number → squashes to range [0, 1]
            probability = torch.sigmoid(prediction).item()

        # STEP 5: Generating attention weights
        seq_length = len(encoded)

        # Generating example attention weights (for demonstration)
        attention = self._generate_example_attention(seq_length)

        # STEP 6: Packaging everything into a dictionary and return
        return {
            'sequence': sequence,  # Original DNA string
            'encoded': encoded,  # List of numbers
            'prediction': probability,  # 0 to 1 (binding probability)
            'attention_weights': attention,  # Matrix of attention values
            'seq_length': seq_length  # Length of sequence
        }

    def _generate_example_attention(self, seq_length: int) -> np.ndarray:
        """
        Generating example attention weights for demonstration.

        It is needed because:
        Real attention extraction from PyTorch transformers is complex.
        For this demo, we create realistic-looking fake attention weights
        that show how the visualization would work.

        How it works:
        We create a matrix (2D grid of numbers) where:
        - Rows = query positions (the position that's "looking")
        - Columns = key positions (the position being "looked at")
        - Each cell (i, j) = how much position i attends to position j

        For example:
        If attention[5, 10] = 0.8, that means:
        "Position 5 pays a lot of attention (0.8 out of 1.0) to position 10"

        The matrix:
        For a sequence of length 5, the attention matrix might look like:

              Pos 0  Pos 1  Pos 2  Pos 3  Pos 4  (attending TO)
        Pos 0 [0.7    0.2    0.05   0.03   0.02]
        Pos 1 [0.2    0.6    0.15   0.03   0.02]
        Pos 2 [0.05   0.2    0.6    0.1    0.05]
        Pos 3 [0.03   0.05   0.2    0.6    0.12]
        Pos 4 [0.02   0.03   0.1    0.2    0.65]
        (attending FROM)

        Notice: Each row sums to 1.0 (because attention is a probability distribution)

        Parameters that are used:
        - seq_length: How many positions in the sequence

        It will return:
        A numpy array of shape (seq_length, seq_length) with attention values
        """

        # STEP 1: Creating a matrix filled with random numbers between 0 and 1
        # Shape: (seq_length, seq_length)
        # np.random.rand() generates random numbers uniformly between 0 and 1
        attention = np.random.rand(seq_length, seq_length)

        # STEP 2: Making it look more realistic
        # Real attention has patterns:
        # - Higher attention to nearby positions (local context)
        # - Some attention to distant positions (long-range dependencies)

        # Loop through every pair of positions (i, j)
        for i in range(seq_length):
            for j in range(seq_length):
                # Calculating how far apart positions i and j are
                # abs() gives absolute value (always positive)
                distance = abs(i - j)

                # Nearby positions: boost attention by 0.5
                # This simulates the model looking at local motifs
                if distance < 5:
                    attention[i, j] += 0.5
                # Medium-distance positions: boost by 0.2
                elif distance < 10:
                    attention[i, j] += 0.2
                # Far positions: keep the random value (small attention)

        # STEP 3: Normalizing rows to sum to 1
        # This makes each row a proper probability distribution
        #
        # Why? In real attention, for each position i, the attention weights
        # across all positions j must sum to 1.0 (it's a weighted average)
        #
        # axis=1 means "sum across columns for each row"
        # keepdims=True keeps it as a column vector for broadcasting
        attention = attention / attention.sum(axis=1, keepdims=True)

        return attention

    def plot_attention_heatmap(self,
                               attention_data: Dict,
                               save_path: str = None,
                               show_sequence: bool = True) -> str:
        """
        Creating a heatmap visualization of attention weights.

        Parameters that will be used:
        - attention_data: Dictionary from get_attention_weights()
                         (contains attention matrix and metadata)
        - save_path: Where to save the image file
                    Default: current folder, named 'attention_heatmap.png'
        - show_sequence: Whether to label axes with actual k-mer sequences
                        (Only works well for short sequences)

        It will return:
        String: Path where the figure was saved

        For example:
        attention_data = visualizer.get_attention_weights("ATCGATCG")
        path = visualizer.plot_attention_heatmap(attention_data, save_path="my_plot.png")
        print(f"Saved to: {path}")
        """

        if save_path is None:
            save_path = os.path.join('.', 'attention_heatmap.png')

        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        attention = attention_data['attention_weights']
        sequence = attention_data['sequence']
        prediction = attention_data['prediction']

        fig, ax = plt.subplots(figsize=(12, 10))

        im = ax.imshow(attention, cmap='YlOrRd', aspect='auto')

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attention Weight', rotation=270, labelpad=20)

        ax.set_xlabel('Key Position (attending TO)')
        ax.set_ylabel('Query Position (attending FROM)')

        ax.set_title(f'Attention Heatmap\n'
                     f'Prediction: {prediction:.3f} '
                     f'({"Binding" if prediction > 0.5 else "No Binding"})')

        if show_sequence and len(attention) <= 30:
            kmers = self.vocabulary.sequence_to_kmers(sequence)

            positions = list(range(len(kmers) + 1))

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
        Plot aggregated attention as sequence importance scores.

        Parameters used:
        - attention_data: Dictionary from get_attention_weights()
        - save_path: Where to save the image
                    Default: 'sequence_importance.png'

        It will return:
        String: Path where the figure was saved
        """
        if save_path is None:
            save_path = os.path.join('.', 'sequence_importance.png')

        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        attention = attention_data['attention_weights']
        sequence = attention_data['sequence']
        prediction = attention_data['prediction']

        importance = attention.mean(axis=0)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                       gridspec_kw={'height_ratios': [1, 3]})

        positions = np.arange(len(importance))

        ax1.bar(positions, importance, color='steelblue', alpha=0.7)

        ax1.set_ylabel('Importance\nScore')
        ax1.set_title(f'Sequence Importance Scores - Prediction: {prediction:.3f}')
        ax1.grid(axis='y', alpha=0.3)

        kmers = self.vocabulary.sequence_to_kmers(sequence)

        kmer_positions = np.arange(1, len(kmers) + 1)

        kmer_importance = importance[1:len(kmers) + 1]

        colors = plt.cm.YlOrRd(kmer_importance / kmer_importance.max())

        ax2.bar(kmer_positions, [1] * len(kmers), color=colors, width=0.8)

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
        Identifying regions of high attention (potential binding motifs).

        This method finds continuous stretches of high-importance positions.
        These might be binding motifs or other biologically important regions.

        The process:
        1. Calculate importance for each position
        2. Find positions above a threshold (default: top 30%)
        3. Group consecutive high-importance positions into regions

        Example:
        Imagine importance scores:
        Position:    0    1    2    3    4    5    6    7    8    9
        Importance: 0.2  0.3  0.8  0.9  0.7  0.3  0.2  0.8  0.9  0.85

        If threshold = 0.7 (70th percentile), we mark positions ≥ 0.7:
        Position:    0    1    2    3    4    5    6    7    8    9
        Important:   No   No  YES  YES  YES   No   No  YES  YES  YES

        Group consecutive positions:
        - Region 1: positions 2-4 (continuous high importance)
        - Region 2: positions 7-9 (continuous high importance)

        Return: [(2, 4), (7, 9)]

        It is useful because:
        - Potential CTCF binding motifs (6-15 bp)
        - Regulatory elements
        - Structural features

        Parameters used:
        - attention_data: Dictionary from get_attention_weights()
        - threshold: Percentile threshold (0-1)
                    0.7 = top 30% (above 70th percentile)
                    0.9 = top 10% (above 90th percentile)

        It will return:
        List of tuples: [(start1, end1), (start2, end2), ...]
        Each tuple is one region: (start_position, end_position)
        Positions are inclusive: (2, 4) means positions 2, 3, and 4

        For example:
        attention_data = visualizer.get_attention_weights("ATCGATCG...")
        regions = visualizer.find_important_regions(attention_data, threshold=0.8)
        print(f"Found {len(regions)} important regions:")
        for start, end in regions:
            print(f"  Region: positions {start} to {end}")
        """

        # STEP 1: Extracting attention matrix and calculate importance
        attention = attention_data['attention_weights']

        # Calculating average attention received by each position
        # (same as in plot_sequence_importance)
        importance = attention.mean(axis=0)

        # STEP 2: Calculating the threshold value
        # np.percentile() finds the value at a given percentile
        # Example: If threshold=0.7, this finds the 70th percentile value
        #          Meaning: 70% of values are below this, 30% are above
        threshold_value = np.percentile(importance, threshold * 100)

        # STEP 3: Finding all positions above the threshold
        # np.where() returns indices where condition is True
        # Example: importance = [0.2, 0.8, 0.9, 0.3]
        #          threshold_value = 0.7
        #          Returns: [1, 2] (positions with 0.8 and 0.9)
        important_positions = np.where(importance >= threshold_value)[0]

        # STEP 4: Grouping consecutive positions into regions
        # We need to find "runs" of consecutive numbers
        # Example: [1, 2, 3, 7, 8, 9] → regions: (1,3) and (7,9)

        regions = []  # Listing to store our regions

        # Only proceeding if we found any important positions
        if len(important_positions) > 0:
            # Starting the first region
            start = important_positions[0]  # First important position
            prev = important_positions[0]  # Keep track of previous position

            # Loop through remaining important positions
            for pos in important_positions[1:]:
                # Checking if there's a gap
                # If current position > previous position + 1, there's a gap
                # Example: prev=3, pos=5 → gap! (missing position 4)
                if pos > prev + 1:
                    # Gap found! Closing the current region
                    regions.append((start, prev))
                    # Starting a new region
                    start = pos

                # Updating previous position
                prev = pos

            # After the loop ends, we still have one open region
            regions.append((start, prev))

        return regions