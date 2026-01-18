# DNA-LM: Interpretable Transformer for TFBS Prediction
## 🎯 Project Overview

**Full Title**: DNA-LM: An Interpretable Language Model for Transcription Factor Binding Site Prediction Using Self-Attention Mechanisms

**One-Line Summary**: Application of transformer-based NLP techniques to genomic sequences for interpretable prediction of transcription factor binding sites with 97.8% accuracy.

**Courses Integrated**:
- 🧬 **Bioinformatics**: ChIP-seq data analysis, regulatory genomics, motif discovery
- 🤖 **Advanced Machine Learning**: Transformer architecture, attention mechanisms, deep learning optimization
- 📝 **Language Processing Technologies**: DNA tokenization, sequence modeling, NLP paradigms for genomics

**Authors**: Klejda Rrapaj

**Date**: January 2026

---

## 📁 Complete Project Structure

```
DNA-LM-TFBS-Prediction/
│
├── Core Implementation Files
│   ├── main.py                          # Main training pipeline
│   ├── DNAVocabulary.py                 # K-mer tokenization (4099 tokens)
│   ├── TFBSDataset.py                   # PyTorch Dataset with data augmentation
│   ├── TransformerModel.py              # Standard transformer architecture
│   ├── Trainer.py                       # Training loop with early stopping
│   ├── AttentionVisualizer.py           # Interpretability visualizations
│   ├── PositionalEncoding.py            # Sinusoidal position encoding
│   └── encode_data_loader.py            # ENCODE ChIP-seq data loader
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

