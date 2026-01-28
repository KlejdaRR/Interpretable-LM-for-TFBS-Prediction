
"""
Transformer Model for TFBS Prediction

This module implements a transformer model using standard PyTorch components.

What is a Transformer?
A transformer is a type of neural network architecture that was invented for language processing
but works great for DNA sequences too! It uses "attention" to focus on important parts.

Key components:
- Embedding: Converting token IDs (numbers) to vectors (lists of numbers)
  Example: Token 42 → [0.1, -0.3, 0.7, 0.2, ...] (128 numbers)

- Positional Encoding: Adding position information
  Why? Transformers don't naturally know the order of tokens
  We add special patterns that tell it "this is position 1, this is position 2..."

- Attention: The model learns which parts of the sequence are important
  It can look at position 50 and decide "position 10 is relevant to understanding this"

- Classification Head: Final layer that makes the prediction
  "Based on everything I've seen, this DNA binds CTCF" (output: 0.95 = 95% confidence)

Why Transformers for DNA?
1. They capture long-range dependencies (position 10 can influence position 150)
2. They process all positions in parallel (faster than RNNs)
3. Attention weights are interpretable (we can see what the model focuses on)
4. They work well with sequential data (DNA is a sequence)
"""

import torch.nn as nn
import math
from models.PositionalEncoding import PositionalEncoding


class TransformerModel(nn.Module):
    """
    A transformer model for predicting transcription factor binding sites.

    Architecture Overview:
    ---------------------
    Analogy: a pipeline with 4 stages:

    DNA Sequence → [1] → [2]         →       [3]     →    [4] → Prediction
                   ↓     ↓                   ↓            ↓
              Embedding  Pos.Encoding        Transformer  Classification


    1. Embedding Layer: Converts token IDs → dense vectors
       Input:  [2, 42, 137, 203, ...]  (token IDs)
       Output: [[0.1, -0.3, ...], [0.5, 0.2, ...], ...]  (vectors)

    2. Positional Encoding: Adds position information
       Tells the model "this vector is at position 1, that one at position 2"

    3. Transformer Encoder: Learns patterns using attention
       Each position looks at other positions and decides what's important
       Does this 4 times (4 layers) to learn increasingly complex patterns

    4. Classification Head: Makes final prediction
       Takes the processed information → outputs binding probability

    Example flow with actual numbers:
    Input:  DNA "ATCGATCG" → tokens [2, 42, 137, 203]
    Embed:  Each token → 128-dimensional vector
    Trans:  Attention learns "position 2 and 3 are important together"
    Class:  Output 0.95 → 95% chance of CTCF binding
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

        What happens here:
        We're building the model architecture - creating all the layers and components.

        Parameters explained:
            vocab_size: Size of the vocabulary (number of unique tokens)
                       For k=6, this is 4^6 + 3 = 4,099
                       Why? 4,096 possible k-mers + 3 special tokens (PAD, UNK, CLS)

            d_model: Dimension of embeddings (how many numbers represent each k-mer)
                    Default: 128
                    Each token gets converted to a vector of 128 numbers

            nhead: Number of attention heads
                  Default: 8
                  Example: d_model=128, nhead=8 → each head gets 128/8 = 16 dimensions

            num_layers: Number of transformer layers (how many times to process)
                       Default: 4
                       Each layer refines the understanding.
                       More layers = more complex patterns but harder to train

            dim_feedforward: Size of feedforward network inside each transformer layer
                            Default: 512

            dropout: Dropout rate for regularization
                    Default: 0.1 (10%)
                    Randomly "turns off" 10% of neurons during training
                    Why? Prevents overfitting (memorizing training data)
                    Forces the model to learn robust patterns

            max_seq_length: Maximum sequence length the model can handle
                           Default: 200
                           Our DNA sequences are 200bp (base pairs)

        What gets created:
        - self.embedding: The embedding layer (vocab → vectors)
        - self.pos_encoder: Adds position information
        - self.transformer_encoder: The main transformer (attention + feedforward)
        - self.classifier: Final prediction layer
        """
        super().__init__()

        self.d_model = d_model
        self.model_name = "Transformer TFBS Predictor"

        # LAYER 1: EMBEDDING LAYER - Converts token IDs (integers) to dense vectors (lists of numbers)

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,  # How many tokens we have (4,099)
            embedding_dim=d_model,      # How many numbers per token (128)
            padding_idx=0               # Token ID 0 is PAD, it should be ignored
        )

        # LAYER 2: POSITIONAL ENCODING - Adds information about position in the sequence
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)

        # LAYER 3: TRANSFORMER ENCODER

        # First, defining what ONE transformer layer looks like
        # We'll stack multiple copies of this to make the full transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,              # Input/output dimension (128)
            nhead=nhead,                  # Number of attention heads (8)
            dim_feedforward=dim_feedforward,  # Size of feedforward network (512)
            dropout=dropout,              # Dropout rate (0.1)
            batch_first=True,             # Input shape: (batch, sequence, features)
            activation='relu'             # Activation function (ReLU = max(0, x))
        )

        # What's inside ONE encoder layer?
        # 1. Multi-head self-attention
        #    - Each position looks at all other positions
        #    - Learns which positions are relevant to each other
        # 2. Add & Normalize
        #    - Adds input to output (residual connection)
        #    - Normalizes to keep values stable
        # 3. Feedforward network
        #    - Two linear layers with ReLU in between
        #    - Processes each position independently
        # 4. Add & Normalize again

        # Now, stacking multiple encoder layers (default: 4)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,    # The blueprint we defined above
            num_layers=num_layers  # How many copies to stack (4)
        )

        # Visualization of stacked layers:
        # Input → Layer 1 → Layer 2 → Layer 3 → Layer 4 → Output
        #         (basic    (motifs)  (inter-   (high-
        #         patterns)           actions)  level)

        # LAYER 4: CLASSIFICATION HEAD - Takes the transformer output and makes the final prediction
        # Architecture of classification head:
        # Input (128) → Linear → ReLU → Dropout → Linear → Output (1)
        #               (64)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            # 128 // 2 = 64
            # Reducing dimensions
            nn.ReLU(),
            nn.Dropout(dropout),     # Regularization (10% dropout)
            nn.Linear(d_model // 2, 1)    # Final output: single number
        )

        self._init_weights()

        print(f"\n{self.model_name} created:")
        print(f"  - Embedding dimension (d_model): {d_model}")
        print(f"  - Number of attention heads: {nhead}")
        print(f"  - Number of transformer layers: {num_layers}")

    def _init_weights(self):
        """
        Initializing model weights with good starting values.
        """

        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)

        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                # Using Xavier (Glorot) uniform initialization for weights
                nn.init.xavier_uniform_(module.weight)

                # Setting biases to zero (if they exist)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids, return_attention=False):
        """
        Forward pass through the model.

        What happens step by step:
        1. Convert token IDs to embeddings (numbers → vectors)
        2. Add positional information (tell model where each token is)
        3. Create mask for padding (tell model to ignore PAD tokens)
        4. Pass through transformer (learn patterns with attention)
        5. Extract CLS token (use first position as sequence summary)
        6. Make final prediction (classify as binding or non-binding)

        Visual flow:
        Input IDs: [2, 42, 137, 203, 0, 0]
                   ↓ [Step 1: Embedding]
        Vectors:   [[0.1, -0.3, ...], [0.5, 0.2, ...], ...]
                   ↓ [Step 2: Add position]
        Positioned: [[0.1, -0.2, ...], [0.5, 0.3, ...], ...]
                   ↓ [Step 3: Mask padding]
        Mask:      [False, False, False, False, True, True]
                   ↓ [Step 4: Transformer]
        Encoded:   [[0.8, 0.3, ...], [0.2, -0.1, ...], ...]
                   ↓ [Step 5: Take CLS]
        CLS:       [0.8, 0.3, ...]
                   ↓ [Step 6: Classify]
        Output:    0.95 (95% chance of binding)

        Parameters used:
            input_ids: Tensor of shape (batch_size, seq_length)
                      Contains token IDs (integers)
                      Example shape: (32, 200) = 32 sequences of length 200
                      Example content: [[2, 42, 137, ...], [2, 93, 512, ...], ...]

            return_attention: If True, return attention weights (for visualization)
                            Default: False (we don't need attention for training)
                            Set to True when you want to see what model focuses on

        It will return:
            predictions: Tensor of shape (batch_size, 1)
                        Raw scores (logits) before sigmoid
                        Example: [[2.3], [-1.5], [0.8], ...]
                        Apply sigmoid to get probabilities:
                        sigmoid(2.3) = 0.91 → 91% binding
                        sigmoid(-1.5) = 0.18 → 18% binding

            attention_weights: (optional, if return_attention=True)
                              Shows which positions the model focused on
                              Used for interpretability and visualization
        """
        # STEP 1: EMBEDDING
        # Converting token IDs to embedding vectors
        x = self.embedding(input_ids)

        # Scaling embeddings by sqrt(d_model)
        x = x * math.sqrt(self.d_model)

        # STEP 2: ADDING POSITIONAL ENCODING
        x = self.pos_encoder(x)

        # STEP 3: CREATING PADDING MASK
        # Telling the transformer to ignore PAD tokens (ID = 0)
        padding_mask = (input_ids == 0)

        # STEP 4: TRANSFORMER ENCODING
        x = self.transformer_encoder(
            x,
            src_key_padding_mask=padding_mask
        )
        # STEP 5: EXTRACT CLS TOKEN
        cls_output = x[:, 0, :]

        # STEP 6: CLASSIFICATION
        predictions = self.classifier(cls_output)

        if return_attention:
            attention_weights = self.get_real_attention()
            return predictions, attention_weights

        return predictions