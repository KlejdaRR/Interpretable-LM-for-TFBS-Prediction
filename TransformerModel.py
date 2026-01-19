
"""
Transformer Model for TFBS Prediction

This module implements a transformer model using standard PyTorch components.

What is a Transformer?
A transformer is a type of neural network architecture that was invented for language processing
but works great for DNA sequences too! It uses "attention" to focus on important parts.

Analogy:
- When you read a sentence, you pay more attention to important words
- When the transformer reads DNA, it pays more attention to important positions
- It learns which positions matter most for binding

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
4. They work well with sequential data (DNA is a sequence, just like text!)
"""

import torch.nn as nn
import math
from PositionalEncoding import PositionalEncoding


class TransformerModel(nn.Module):
    """
    A transformer model for predicting transcription factor binding sites.

    Architecture Overview:
    ---------------------
    Analogy: a pipeline with 4 stages:

    DNA Sequence → [1] → [2] → [3] → [4] → Prediction
                   ↓     ↓     ↓     ↓
              Embedding Pos  Trans- Classi-
                       Enc.  former  fier

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
        Think of it like assembling a machine: we're putting together all the parts
        before we actually use it to make predictions.

        Parameters explained:
            vocab_size: Size of the vocabulary (number of unique tokens)
                       For k=6, this is 4^6 + 3 = 4,099
                       Why? 4,096 possible k-mers + 3 special tokens (PAD, UNK, CLS)

            d_model: Dimension of embeddings (how many numbers represent each k-mer)
                    Default: 128
                    Each token gets converted to a vector of 128 numbers
                    Smaller (64): Faster, less memory, less expressive
                    Larger (256): Slower, more memory, more expressive
                    128 is a good balance

            nhead: Number of attention heads
                  Default: 8
                  Think: The model has 8 different "perspectives" to look at the sequence
                  - Head 1 might focus on local patterns (nearby positions)
                  - Head 2 might focus on long-range patterns (distant positions)
                  - Head 3 might focus on specific motifs
                  IMPORTANT: nhead must divide d_model evenly
                  Example: d_model=128, nhead=8 → each head gets 128/8 = 16 dimensions

            num_layers: Number of transformer layers (how many times to process)
                       Default: 4
                       Each layer refines the understanding:
                       - Layer 1: Learns basic patterns (dinucleotides)
                       - Layer 2: Learns motifs (6-mers)
                       - Layer 3: Learns complex interactions
                       - Layer 4: Learns high-level features
                       More layers = more complex patterns but harder to train

            dim_feedforward: Size of feedforward network inside each transformer layer
                            Default: 512
                            This is like a "mini brain" inside each layer
                            Typically 2-4x larger than d_model

            dropout: Dropout rate for regularization
                    Default: 0.1 (10%)
                    Randomly "turns off" 10% of neurons during training
                    Why? Prevents overfitting (memorizing training data)
                    Forces the model to learn robust patterns
                    Range: 0.0 (no dropout) to 0.5 (aggressive dropout)

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

        # Storing model dimension for later use
        self.d_model = d_model

        # Giving our model a friendly name
        self.model_name = "Transformer TFBS Predictor"

        # LAYER 1: EMBEDDING LAYER
        # What does this do?
        # Converts token IDs (integers) to dense vectors (lists of numbers)
        #
        # Why do we need this?
        # Neural networks can't process integers directly, they need vectors
        # Think: Converting words to meaning
        # - "cat" and "dog" are similar → their vectors should be similar
        # - "cat" and "car" are different → their vectors should be different
        #
        # Example:
        # Token ID 42 → [0.1, -0.3, 0.7, 0.2, -0.5, ..., 0.9]  (128 numbers)
        # Token ID 137 → [0.4, 0.2, -0.1, 0.6, 0.3, ..., -0.2]  (128 numbers)
        #
        # These vectors are learned during training!
        # Initially random, but the model adjusts them to group similar k-mers
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,  # How many tokens we have (4,099)
            embedding_dim=d_model,      # How many numbers per token (128)
            padding_idx=0               # Token ID 0 is PAD, it should be ignored
            # The embedding for PAD stays at zero
        )

        # LAYER 2: POSITIONAL ENCODING
        # What does this do?
        # Adds information about position in the sequence
        #
        # Why do we need this?
        # Transformers process all positions in parallel (unlike RNNs)
        # Without positional encoding, the model can't tell if "ATCG" is at
        # position 1 or position 100!
        #
        # How does it work?
        # It adds special patterns (sine and cosine waves) to each position
        # Position 1 gets pattern A, position 2 gets pattern B, etc.
        # These patterns are designed so the model can learn relative positions
        #
        # Example:
        # Before: [0.1, -0.3, 0.7, 0.2]  (just the embedding)
        # After:  [0.1, -0.2, 0.8, 0.3]  (embedding + position info)
        #         ↑ same  ↑ changed by adding position pattern
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)

        # LAYER 3: TRANSFORMER ENCODER
        # This is the heart of the model - where the magic happens!

        # First, defining what ONE transformer layer looks like
        # Think of this as a blueprint for a single processing unit
        # We'll stack multiple copies of this to make the full transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,              # Input/output dimension (128)
            nhead=nhead,                  # Number of attention heads (8)
            dim_feedforward=dim_feedforward,  # Size of feedforward network (512)
            dropout=dropout,              # Dropout rate (0.1)
            batch_first=True,             # Input shape: (batch, sequence, features)
            # This makes it easier to work with
            activation='relu'             # Activation function (ReLU = max(0, x))
            # Adds non-linearity so model can learn complex patterns
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

        # Now, stacking multiple encoder layers
        # We'll stack num_layers copies (default: 4)
        # Each layer refines the understanding from the previous layer
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,    # The blueprint we defined above
            num_layers=num_layers  # How many copies to stack (4)
        )

        # Visualization of stacked layers:
        # Input → Layer 1 → Layer 2 → Layer 3 → Layer 4 → Output
        #         (basic    (motifs)  (inter-   (high-
        #         patterns)           actions)  level)

        # LAYER 4: CLASSIFICATION HEAD
        # What does this do?
        # Takes the transformer output and makes the final prediction
        # "Does this DNA sequence bind CTCF?" → Yes (1) or No (0)
        #
        # Why nn.Sequential?
        # It chains multiple layers together in order
        # Data flows through each layer one after another
        #
        # Architecture of classification head:
        # Input (128) → Linear → ReLU → Dropout → Linear → Output (1)
        #               (64)
        #
        # Step by step:
        # 1. Linear(128 → 64): Reduce dimensions, compress information
        # 2. ReLU: Add non-linearity, keep positive values
        # 3. Dropout: Randomly turn off neurons (prevent overfitting)
        # 4. Linear(64 → 1): Final prediction (single number)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),  # // means integer division
            # 128 // 2 = 64
            # Reducing dimensions
            nn.ReLU(),                          # Activation function
            # Keeps positive values, zeros out negatives
            nn.Dropout(dropout),                # Regularization (10% dropout)
            nn.Linear(d_model // 2, 1)          # Final output: single number
            # This will be the binding score
        )

        # Why use the output as-is?
        # The raw output might be any number: -2.5, 0.0, 3.7, etc.
        # We'll apply sigmoid later to convert to probability: 0.0 to 1.0
        # sigmoid(3.7) = 0.976 → 97.6% chance of binding

        # Initializing weights properly
        # Random initialization matters! Good initialization helps training
        self._init_weights()

        # Calculating total parameters
        # Parameters = numbers the model learns during training
        # More parameters = more capacity but also more data needed
        total_params = sum(p.numel() for p in self.parameters())
        # p.numel() counts elements in each parameter tensor
        # We sum across all parameters to get total

        # Trainable parameters (should be same as total for this model)
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # requires_grad=True means "update this during training"

        print(f"\n{self.model_name} created:")
        print(f"  - Embedding dimension (d_model): {d_model}")
        print(f"  - Number of attention heads: {nhead}")
        print(f"  - Number of transformer layers: {num_layers}")
        print(f"  - Total parameters: {total_params:,}")  # :, adds commas (1,326,081)
        print(f"  - Trainable parameters: {trainable_params:,}")

    def _init_weights(self):
        """
        Initializing model weights with good starting values.

        Why do we need this?
        Neural networks need good initial weights to train successfully.
        - Too large: Training explodes (gradients blow up)
        - Too small: Training stalls (gradients vanish)
        - Random but smart: Training converges smoothly

        What does this do?
        1. Initialize embeddings with small random values
        2. Initialize classifier weights with Xavier initialization
        3. Set biases to zero
        """
        # Initializing embeddings
        # We use normal distribution with mean=0, std=0.02
        # This gives us small random values centered around zero
        # Example: 0.01, -0.03, 0.02, -0.01, ...
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        # The underscore (_) means "modify in-place" (change the tensor directly)

        # Initializing classifier layers
        # We loop through each layer in the classifier
        for module in self.classifier.modules():
            # Checking if this is a linear layer
            # isinstance() checks the type: "Is this a Linear layer?"
            if isinstance(module, nn.Linear):
                # Using Xavier (Glorot) uniform initialization for weights
                # This helps with training stability
                nn.init.xavier_uniform_(module.weight)

                # Setting biases to zero (if they exist)
                # Not all layers have biases, so we check first
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids, return_attention=False):
        """
        Forward pass through the model.

        The forward() method defines what happens during this pass.
        When we call model(input_ids), PyTorch automatically calls forward().

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

        Example:
            # During training (don't need attention)
            predictions = model(input_ids)  # Shape: (32, 1)

            # During visualization (want to see attention)
            predictions, attention = model(input_ids, return_attention=True)
        """
        # STEP 1: EMBEDDING
        # Converting token IDs to embedding vectors
        # Input shape:  (batch_size, seq_length)
        # Output shape: (batch_size, seq_length, d_model)
        #
        # Example:
        # Input:  [[2, 42, 137], [2, 93, 512]]  (2 sequences, length 3)
        # Output: [[[0.1, -0.3, ...], [0.5, 0.2, ...], [0.8, -0.1, ...]],
        #          [[0.1, -0.3, ...], [0.3, 0.4, ...], [0.6, 0.1, ...]]]
        #         (2 sequences, length 3, 128 dimensions)
        x = self.embedding(input_ids)

        # Scaling embeddings by sqrt(d_model)
        # Why? This is a standard practice in transformers
        # It helps maintain gradient magnitudes
        # sqrt(128) ≈ 11.3, so we multiply all embeddings by 11.3
        x = x * math.sqrt(self.d_model)

        # STEP 2: ADDING POSITIONAL ENCODING
        # Adding position information to embeddings
        # The positional encoder adds sine/cosine patterns
        # Shape stays the same: (batch_size, seq_length, d_model)
        #
        # Before: [0.1, -0.3, 0.7, 0.2]  (just embedding)
        # After:  [0.1, -0.2, 0.8, 0.3]  (embedding + position pattern)
        x = self.pos_encoder(x)

        # STEP 3: CREATING PADDING MASK
        # Telling the transformer to ignore PAD tokens (ID = 0)
        #
        # Why do we need this?
        # Sequences have different lengths, but batches need uniform size
        # Short sequences are padded with PAD tokens (ID=0)
        # We don't want the model to learn from padding!
        #
        # Example:
        # input_ids = [[2, 42, 137, 203, 0, 0],
        #              [2, 93, 512, 42, 137, 0]]
        #
        # padding_mask = [[False, False, False, False, True, True],
        #                 [False, False, False, False, False, True]]
        #
        # True means "ignore this position"
        # False means "pay attention to this position"
        padding_mask = (input_ids == 0)
        # This creates a boolean tensor: True where input_ids is 0, False otherwise

        # STEP 4: TRANSFORMER ENCODING
        # Passing through the transformer encoder
        # This is where attention happens!
        # Each position looks at other positions and learns patterns
        #
        # What happens inside:
        # - Multi-head attention: Each head focuses on different patterns
        # - Feedforward: Process information at each position
        # - Residual connections: Add input to output (helps training)
        # - Layer normalization: Keep values stable
        #
        # This happens 4 times (4 layers), each time refining the representation
        x = self.transformer_encoder(
            x,                              # Input: (batch, seq, features)
            src_key_padding_mask=padding_mask  # Mask: (batch, seq)
        )
        # Output shape: (batch_size, seq_length, d_model)
        # Same shape as input, but now with learned patterns!

        # STEP 5: EXTRACT CLS TOKEN
        # Using the first token (CLS) as the sequence representation
        #
        # Why the first token?
        # The CLS token is special - it's designed to aggregate information
        # During attention, CLS looks at all other positions and summarizes them
        # Think: CLS is like a team captain who knows what everyone is doing
        #
        # Indexing explanation:
        # x has shape: (batch_size, seq_length, d_model)
        # x[:, 0, :] means:
        #   : = all batches
        #   0 = first position (the CLS token)
        #   : = all features
        #
        # Example:
        # x.shape = (32, 200, 128)  (32 sequences, 200 tokens, 128 features)
        # cls_output.shape = (32, 128)  (32 sequences, 128 features)
        cls_output = x[:, 0, :]

        # STEP 6: CLASSIFICATION
        # Making final prediction using the classification head
        #
        # What happens:
        # CLS (128) → Linear (64) → ReLU → Dropout → Linear (1) → prediction
        #
        # Input:  [0.8, 0.3, -0.1, ...]  (128 numbers)
        # Output: [2.3]  (1 number, the raw score)
        #
        # Shape: (batch_size, d_model) → (batch_size, 1)
        # Example: (32, 128) → (32, 1)
        predictions = self.classifier(cls_output)

        # Returning results
        # If attention was requested, we'd return it here
        # For now, we return None for attention (we'd need to extract it from transformer)
        if return_attention:
            # In a more advanced implementation, we'd extract attention weights here
            # For now, returning None as a placeholder
            return predictions, None

        # Normal case: just return predictions
        return predictions