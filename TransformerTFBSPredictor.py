import torch.nn as nn
import CNNMotifExtractor as CNNMotifExtractor
import PositionalEncoding as PositionalEncoding

class TransformerTFBSPredictor(nn.Module):
    """
    Complete DNA-LM model: CNN + Transformer for TFBS prediction.

    Architecture:
    1. Embedding layer: token → vector
    2. CNN layer: local motif detection
    3. Positional encoding: position information
    4. Transformer layers: long-range dependencies
    5. Classification head: binding site prediction
    """

    def __init__(self, vocab_size: int, embedding_dim: int = 128,
                 num_heads: int = 8, num_layers: int = 4,
                 dim_feedforward: int = 512, dropout: float = 0.1,
                 max_length: int = 200):
        super().__init__()

        # 1. Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # 2. CNN for motif detection
        self.cnn = CNNMotifExtractor(embedding_dim, num_filters=128)
        cnn_output_dim = self.cnn.output_dim

        # Project CNN output to transformer dimension
        self.cnn_projection = nn.Linear(cnn_output_dim, embedding_dim)

        # 3. Positional encoding
        self.pos_encoder = PositionalEncoding(embedding_dim, max_length, dropout)

        # 4. Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 5. Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 1)
        )

        # For attention visualization
        self.attention_weights = None

    def forward(self, input_ids, return_attention: bool = False):
        """
        Forward pass.

        Args:
            input_ids: (batch_size, seq_len)
            return_attention: Whether to return attention weights

        Returns:
            predictions: (batch_size, 1)
            attention_weights: Optional attention weights for interpretability
        """
        # 1. Embedding
        x = self.embedding(input_ids)  # (batch, seq_len, embedding_dim)

        # 2. CNN motif detection
        cnn_features = self.cnn(x)  # (batch, seq_len, cnn_output_dim)
        x = self.cnn_projection(cnn_features)  # (batch, seq_len, embedding_dim)

        # 3. Add positional encoding
        x = self.pos_encoder(x)

        # 4. Create attention mask for padding
        padding_mask = (input_ids == 0)

        # 5. Transformer encoding
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # 6. Use CLS token representation for classification
        cls_representation = x[:, 0, :]  # (batch, embedding_dim)

        # 7. Classification
        output = self.classifier(cls_representation)  # (batch, 1)

        if return_attention:
            # Extract attention weights for interpretability
            # This would require modifying the transformer to return attention
            return output, None

        return output
