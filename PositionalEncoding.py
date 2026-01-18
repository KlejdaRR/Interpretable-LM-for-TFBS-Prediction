import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Adds position information to embeddings using sinusoidal functions.

    Why do we need this?
    Transformers process all positions simultaneously (unlike RNNs which process
    sequentially). Without positional encoding, the model can't tell if "ATG"
    appears at position 5 or position 50!

    How it works:
    ------------
    Uses sine and cosine functions with different frequencies:
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Different positions get unique patterns of sine/cosine values.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initializing positional encoding.

        It will take as arguments:
            d_model: Dimension of embeddings
            max_len: Maximum sequence length we'll ever see
            dropout: Dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Creating a matrix to store positional encodings
        # Shape: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # Position indices: [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Create the division term for the formula
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, but should be saved with model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Adding positional encoding to input embeddings.

        It will take as arguments:
            x: Tensor of shape (batch_size, seq_length, d_model)

        It will return:
            Tensor with positional encoding added
        """
        # Add positional encoding
        # self.pe[:x.size(1)] selects only the needed positions
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)