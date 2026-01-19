# DNA-LM: Interpretable Transformer for TFBS Prediction

**Full Title**: DNA-LM: An Interpretable Language Model for Transcription Factor Binding Site Prediction Using Self-Attention Mechanisms

**One-Line Summary**: Application of transformer-based NLP techniques to genomic sequences for interpretable prediction of transcription factor binding sites with 97.8% accuracy.

**Courses Integrated**:
- **Bioinformatics**: ChIP-seq data analysis, regulatory genomics, motif discovery
- **Advanced Machine Learning**: Transformer architecture, attention mechanisms, deep learning optimization

**Date**: January 2026

---

## Complete Project Structure

```
Interpretable-LM-for-TFBS-Prediction/
│
├── models/                          # Model architectures
│   ├── __init__.py
│   ├── TransformerModel.py         ← Main model
│   └── PositionalEncoding.py       ← Model component
│
├── data/                            # Data processing
│   ├── __init__.py
│   ├── DNAVocabulary.py            ← Tokenization
│   ├── TFBSDataset.py              ← PyTorch Dataset
│   └── encode_data_loader.py       ← Data loading
│
├── training/                        # Training logic
│   ├── __init__.py
│   └── Trainer.py                  ← Training loop
│
├── visualization/                   # Analysis & visualization
│   ├── __init__.py
│   └── AttentionVisualizer.py      ← Attention plots
│
├── data_files/                      # Raw data
│   └── ENCFF308JDD.bed             ← ENCODE data
│
├── outputs/                         # Results
│   ├── best_model.pt
│   └── attention_*.png
│
├── main.py                          # Main script
└── README.md                        # Documentation
```

### Model Architecture
```
DNA Sequence (200 bp)
    ↓
Embedding Layer (4099 → 128 dims)
    ↓
Positional Encoding (sinusoidal)
    ↓
Transformer Encoder ×4
  • Multi-head attention (8 heads)
  • Feed-forward networks
  • Layer normalization
    ↓
Classification Head (CLS token)
    ↓
Prediction: Binding (1) or Non-binding (0)

Total Parameters: 1,326,081
```

## ⚙️ Technical Specifications

### Hyperparameters
```python
# Model
d_model = 128           # Embedding dimension
nhead = 8              # Number of attention heads
num_layers = 4         # Transformer layers
dim_feedforward = 512  # FFN dimension
dropout = 0.1          # Dropout rate

# Training
learning_rate = 1e-4   # Adam optimizer
batch_size = 32        # Training batch size
max_epochs = 20        # Maximum epochs
early_stopping = 5     # Patience for early stopping

# Data
k = 6                  # K-mer size
max_seq_length = 200   # Sequence length
```

**Authors**: Klejda Rrapaj, Sildi Ricku

