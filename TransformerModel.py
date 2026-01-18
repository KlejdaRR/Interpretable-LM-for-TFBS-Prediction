"""
Transformer Model for TFBS Prediction
This module implements a transformer model using standard PyTorch components.

- Transformer: A neural network architecture that uses "attention" to focus on
  important parts of the input
- Embedding: Converting token IDs to dense vectors
- Positional Encoding: Adding position information (transformers don't naturally
  know the order of tokens)
- Attention: The model learns which parts of the sequence are important for prediction
"""

import torch.nn as nn
import math
from PositionalEncoding import PositionalEncoding


class TransformerModel(nn.Module):
    """
    A simple transformer model for TFBS prediction.

    Architecture:
    -------------
    1. Embedding layer: Convert token IDs → vectors
    2. Positional encoding: Add position information
    3. Transformer encoder: Learn patterns in the sequence
    4. Classification head: Make final prediction (binding or not)
    """

    def __init__(self,
                 vocab_size: int,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 4,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 max_seq_length: int = 200):
        """
        Initializing the transformer model.

        It will take as arguments:
            vocab_size: Size of the vocabulary (number of unique k-mers)
            d_model: Dimension of embeddings (how many numbers represent each k-mer)
            nhead: Number of attention heads
                   - More heads = model can focus on different patterns simultaneously
                   - Must divide d_model evenly (e.g., d_model=128, nhead=8 → 16 per head)
            num_layers: Number of transformer layers
                        - More layers = model can learn more complex patterns
                        - But also harder to train and slower
            dim_feedforward: Size of feedforward network in transformer
            dropout: Dropout rate for regularization (prevents overfitting)
            max_seq_length: Maximum sequence length
        """
        super().__init__()

        self.d_model = d_model
        self.model_name = "Transformer TFBS Predictor"

        # LAYER 1: EMBEDDING
        # It converts token IDs to dense vectors
        # For example: token 42 → [0.1, -0.3, 0.7, ..., 0.2] (d_model numbers)
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=0  # Token ID 0 is PAD, therefore it should be ignored
        )

        # LAYER 2: POSITIONAL ENCODING
        # Transformers don't naturally understand position/order
        # We will add special vectors that encode position information
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)

        # LAYER 3: TRANSFORMER ENCODER

        # First, we define what ONE transformer layer looks like
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # Input shape: (batch, seq, feature)
            activation='relu'
        )

        # Stacking multiple layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # LAYER 4: CLASSIFICATION HEAD
        # It takes the transformer output and makes final prediction
        # We'll use the first token (CLS token) for classification
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),  # Reducing dimensions
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)  # Output: single number (binding probability)
        )

        # Initializing weights properly
        self._init_weights()

        # Calculating total parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"\n{self.model_name} created:")
        print(f"  - Embedding dimension (d_model): {d_model}")
        print(f"  - Number of attention heads: {nhead}")
        print(f"  - Number of transformer layers: {num_layers}")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")

    def _init_weights(self):
        """
        Initializing model weights.
        """
        # Initializing embeddings
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)

        # Initializing classifier
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids, return_attention=False):
        """
        Forward pass through the model.

        Steps:
        1. Convert token IDs to embeddings
        2. Add positional information
        3. Create mask for padding tokens
        4. Pass through transformer
        5. Extract CLS token representation
        6. Make final prediction

        It will take as arguments:
            input_ids: Tensor of shape (batch_size, seq_length)
                      Contains token IDs
            return_attention: If True, return attention weights for visualization

        It will return:
            predictions: Tensor of shape (batch_size, 1)
                        Raw scores (apply sigmoid to get probabilities)
            attention_weights: (optional) For interpretability
        """
        # STEP 1: EMBEDDING
        # Converting IDs to vectors and scale
        # Shape: (batch_size, seq_length, d_model)
        x = self.embedding(input_ids) * math.sqrt(self.d_model)

        # STEP 2: ADDING POSITIONAL ENCODING
        x = self.pos_encoder(x)

        # STEP 3: CREATING PADDING MASK
        # Telling the transformer to ignore PAD tokens (ID = 0)
        # Shape: (batch_size, seq_length)
        # True = it will ignore this position, False = it will pay attention to it
        padding_mask = (input_ids == 0)

        # STEP 4: TRANSFORMER ENCODING
        # The transformer will learn to focus on important parts of the sequence
        x = self.transformer_encoder(
            x,
            src_key_padding_mask=padding_mask
        )

        # STEP 5: EXTRACT CLS TOKEN
        # Using the first token (CLS) as the sequence representation
        # Shape: (batch_size, d_model)
        cls_output = x[:, 0, :]

        # STEP 6: CLASSIFICATION
        # Making final prediction
        # Shape: (batch_size, 1)
        predictions = self.classifier(cls_output)

        if return_attention:
            return predictions, None

        return predictions

