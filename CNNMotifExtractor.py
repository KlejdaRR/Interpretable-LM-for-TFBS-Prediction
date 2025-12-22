import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class CNNMotifExtractor(nn.Module):
    """
    CNN layers for local motif detection.

    Biological motivation:
    - TF binding motifs are typically 6-20 bp
    - CNNs act as motif scanners (similar to position weight matrices)
    - Multiple filter sizes capture motifs of different lengths
    """

    def __init__(self, embedding_dim: int, num_filters: int = 128,
                 filter_sizes: List[int] = [6, 8, 12]):
        super().__init__()

        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=fs, padding=fs // 2)
            for fs in filter_sizes
        ])

        self.output_dim = num_filters * len(filter_sizes)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, embedding_dim)
        Returns:
            (batch_size, seq_len, num_filters * len(filter_sizes))
        """
        # Transpose for Conv1d: (batch, channels, length)
        x = x.transpose(1, 2)

        # Apply each convolution and activation
        conv_outputs = [F.relu(conv(x)) for conv in self.convs]

        # Concatenate along channel dimension
        x = torch.cat(conv_outputs, dim=1)

        # Transpose back: (batch, length, channels)
        x = x.transpose(1, 2)

        return x
