"""
This is the main script that combines all components to train and evaluate
the DNA-LM model for TFBS prediction.

Usage:
1. Prepare data (sequences and labels)
2. Configure parameters below
3. Run: python main.py

"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import random
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from DNAVocabulary import DNAVocabulary
from TFBSDataset import TFBSDataset
from TransformerModel import TransformerModel
from Trainer import Trainer
from AttentionVisualizer import AttentionVisualizer

# Import ENCODE data loader
try:
    from encode_data_loader import load_encode_peaks
except ImportError:
    load_encode_peaks = None
    print("Note: encode_data_loader.py not found. Only synthetic data will be available.")


def set_random_seed(seed: int = 42):
    """
    Method that sets random seeds for reproducibility.

    Why?
    Because neural networks use randomness (weight initialization, data shuffling, etc.)
    Setting seeds ensures we get the same results each time we run the code.
    This is crucial for debugging and comparing different approaches.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to: {seed}")


def load_data(data_path: str = None):
    """
    Method that loads TFBS data.

    In practice, this would load ENCODE ChIP-seq data.
    For this demo, we will generate synthetic data.

    Expected data format:
    - sequences: List of DNA sequences (strings)
    - labels: List of binary labels (0 or 1)
              1 = TF binds here, 0 = no binding

    It will take as arguments:
        data_path: Path to data file (if None, generate synthetic data)

    It will return:
        (sequences, labels) tuple
    """
    if data_path is None:
        print("\nGenerating synthetic data for demonstration...")

        def generate_random_sequence(length=200):
            # Generating a random DNA sequence.
            return ''.join(random.choices(['A', 'C', 'G', 'T'], k=length))

        # Generating balanced dataset
        n_samples = 1000
        sequences = [generate_random_sequence() for _ in range(n_samples)]
        labels = [random.randint(0, 1) for _ in range(n_samples)]

        print(f"Generated {n_samples} synthetic sequences")
        print(f"  - Positive (binding): {sum(labels)}")
        print(f"  - Negative (no binding): {n_samples - sum(labels)}")

    else:
        # Load real ENCODE data
        if load_encode_peaks is not None:
            print(f"\nLoading real ENCODE ChIP-seq data from {data_path}...")
            sequences, labels = load_encode_peaks(data_path, max_sequences=1000)

            if not sequences:
                print("\n⚠️  Failed to load ENCODE data. Falling back to synthetic data.")
                return load_data(None)  # Fall back to synthetic
        else:
            print("\n⚠️  encode_data_loader.py not found. Using synthetic data.")
            return load_data(None)  # Fall back to synthetic

    return sequences, labels


def evaluate_model(model, dataloader, device):
    """
    Method that does comprehensive evaluation of a model.

    Metrics used:
    - Accuracy: % of correct predictions
    - Precision: Of predicted positives, how many are actually positive?
    - Recall: Of actual positives, how many did we find?
    - F1: Harmonic mean of precision and recall
    - ROC-AUC: Area under ROC curve (overall discrimination ability)

    It will take as arguments:
        model: The model to evaluate (transformer or baseline)
        dataloader: DataLoader with test data
        device: Device to run on

    It will return:
        Dictionary of metrics
    """
    model.eval()

    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)

            # Get predictions
            outputs = model(input_ids).squeeze()
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).float()

            # Collect results
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_predictions),
        'precision': precision_score(all_labels, all_predictions, zero_division=0),
        'recall': recall_score(all_labels, all_predictions, zero_division=0),
        'f1': f1_score(all_labels, all_predictions, zero_division=0),
        'roc_auc': roc_auc_score(all_labels, all_probabilities)
    }

    return metrics


def print_metrics(metrics: dict, model_name: str):
    """
    Method that prints evaluation metrics in a nice format.
    """
    print(f"\n{'=' * 70}")
    print(f"{model_name} - Evaluation Results")
    print(f"{'=' * 70}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"{'=' * 70}")


def main():
    """
    Main training pipeline.

    Steps:
    1. Load and prepare data
    2. Create vocabulary
    3. Create datasets and dataloaders
    4. Train transformer model
    5. Visualize attention (interpretability)
    """

    print("\n" + "=" * 70)
    print("DNA-LM TRAINING PIPELINE")
    print("Interpretable Transformer for TFBS Prediction")
    print("=" * 70)

    config = {
        # Data
        'data_path': './ENCFF308JDD.bed',
        'max_seq_length': 200,
        'test_split': 0.2,
        'val_split': 0.1,

        # Vocabulary
        'k': 6,  # k-mer size

        # Model
        'd_model': 128,
        'nhead': 8,
        'num_layers': 4,
        'dropout': 0.1,

        # Training
        'batch_size': 32,
        'learning_rate': 1e-4,
        'num_epochs': 20,
        'early_stopping_patience': 5,

        # Paths
        'output_dir': './outputs',  # Directory for saving results

        # Other
        'random_seed': 42,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    print(f"\nOutput directory created: {config['output_dir']}")

    # STEP 1: Set random seed
    set_random_seed(config['random_seed'])

    # STEP 2: Load data
    sequences, labels = load_data(config['data_path'])

    # STEP 3: Create vocabulary
    print("\n" + "-" * 70)
    print("Creating DNA vocabulary...")
    print("-" * 70)
    vocabulary = DNAVocabulary(k=config['k'])

    # STEP 4: Split data (train/val/test)
    print("\n" + "-" * 70)
    print("Splitting data...")
    print("-" * 70)

    n_total = len(sequences)
    n_test = int(n_total * config['test_split'])
    n_val = int((n_total - n_test) * config['val_split'])
    n_train = n_total - n_test - n_val

    # Shuffle and split
    indices = list(range(n_total))
    random.shuffle(indices)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    train_sequences = [sequences[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]

    val_sequences = [sequences[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]

    test_sequences = [sequences[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]

    print(f"Data split:")
    print(f"  Training:   {n_train} samples")
    print(f"  Validation: {n_val} samples")
    print(f"  Test:       {n_test} samples")

    # STEP 5: Create datasets
    print("\n" + "-" * 70)
    print("Creating datasets...")
    print("-" * 70)

    train_dataset = TFBSDataset(
        sequences=train_sequences,
        labels=train_labels,
        vocabulary=vocabulary,
        max_length=config['max_seq_length'],
        use_augmentation=True
    )

    val_dataset = TFBSDataset(
        sequences=val_sequences,
        labels=val_labels,
        vocabulary=vocabulary,
        max_length=config['max_seq_length'],
        use_augmentation=False
    )

    test_dataset = TFBSDataset(
        sequences=test_sequences,
        labels=test_labels,
        vocabulary=vocabulary,
        max_length=config['max_seq_length'],
        use_augmentation=False
    )

    # STEP 6: Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False
    )

    # STEP 7: Create and train transformer model
    print("\n" + "-" * 70)
    print("Creating Transformer model...")
    print("-" * 70)

    transformer_model = TransformerModel(
        vocab_size=vocabulary.vocab_size,
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        max_seq_length=config['max_seq_length']
    )

    # Train
    trainer = Trainer(
        model=transformer_model,
        device=config['device'],
        learning_rate=config['learning_rate']
    )

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['num_epochs'],
        early_stopping_patience=config['early_stopping_patience'],
        save_dir=config['output_dir']  # Pass the output directory
    )

    # Evaluate
    print("\n" + "-" * 70)
    print("Evaluating on test set...")
    print("-" * 70)
    transformer_metrics = evaluate_model(
        transformer_model,
        test_loader,
        config['device']
    )
    print_metrics(transformer_metrics, "Transformer Model")

    # STEP 8: Attention visualization (interpretability)
    print("\n" + "=" * 70)
    print("INTERPRETABILITY: Attention Visualization")
    print("=" * 70)

    visualizer = AttentionVisualizer(transformer_model, vocabulary)

    # Visualize a few test examples
    print("\nGenerating attention visualizations for example sequences...")
    for i in range(min(3, len(test_sequences))):
        seq = test_sequences[i]
        label = test_labels[i]

        print(f"\nExample {i + 1}:")
        print(f"  Sequence: {seq[:50]}...")
        print(f"  True label: {label} ({'Binding' if label == 1 else 'No binding'})")

        attention_data = visualizer.get_attention_weights(seq)
        print(f"  Prediction: {attention_data['prediction']:.3f}")

        # Create visualizations with correct paths
        heatmap_path = os.path.join(config['output_dir'], f'attention_example_{i + 1}_heatmap.png')
        importance_path = os.path.join(config['output_dir'], f'attention_example_{i + 1}_importance.png')

        visualizer.plot_attention_heatmap(
            attention_data,
            save_path=heatmap_path
        )
        visualizer.plot_sequence_importance(
            attention_data,
            save_path=importance_path
        )

        # Find important regions
        regions = visualizer.find_important_regions(attention_data)
        if regions:
            print(f"  Important regions: {regions}")

    # STEP 9: Save final summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\nResults saved in: {config['output_dir']}")
    print("  - best_model.pt: Trained model weights")
    print("  - attention_*.png: Attention visualization plots")
    print("=" * 70)


if __name__ == "__main__":
    main()