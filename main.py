import torch
from torch.utils.data import DataLoader
import numpy as np
import random
import os
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from data.DNAVocabulary import DNAVocabulary
from data.TFBSDataset import TFBSDataset
from models.TransformerModel import TransformerModel
from training.Trainer import Trainer
from visualization.AttentionVisualizer import AttentionVisualizer
from LPT_Implementation.RegexMotifDetector import RegexMotifDetector
from LPT_Implementation.DNAGrammar import DNAGrammar

# Importing ENCODE data loader
try:
    from data.encode_data_loader import load_encode_peaks
except ImportError:
    load_encode_peaks = None
    print("Note: encode_data_loader.py not found.")


def set_random_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to: {seed}")


def generate_patterned_sequences(n_sequences=1000, seq_length=200):
    """
    Generate synthetic DNA sequences with actual biological patterns
    with proper random distribution
    """
    sequences = []
    labels = []

    # Known CTCF binding motif
    ctcf_motifs = ["CCGCGNGGAG", "ACCGCGNGGAG", "CCGCGNGGAGA"]

    def fill_motif(motif):
        return ''.join([random.choice(['A', 'C', 'G', 'T']) if base == 'N' else base for base in motif])

    # Known promoter elements
    tata_motifs = ["TATAAA", "TATATA", "TATAWAW"]
    caat_motifs = ["GGTCAATCT", "GGCCAATCT"]

    print("\nGenerating synthetic data with biological patterns...")

    for i in range(n_sequences):
        # Randomly deciding if this sequence is positive (50% chance)
        is_positive = random.random() < 0.5

        if is_positive:
            # Generating positive sequence
            seq = list(random.choices(['A', 'C', 'G', 'T'], k=seq_length))

            # Inserting CTCF motif (70% of positive sequences)
            if random.random() < 0.7:
                motif_template = random.choice(ctcf_motifs)
                motif = fill_motif(motif_template)
                pos = random.randint(20, seq_length - len(motif) - 20)
                for j, base in enumerate(motif):
                    if pos + j < seq_length:
                        seq[pos + j] = base

            # Inserting TATA box (50% of positive sequences)
            if random.random() < 0.5:
                tata_template = random.choice(tata_motifs)
                tata = tata_template.replace('W', random.choice(['A', 'T']))
                tata_pos = random.randint(10, 40)
                for j, base in enumerate(tata):
                    if tata_pos + j < seq_length:
                        seq[tata_pos + j] = base

            # Inserting E-box (40% of positive sequences)
            if random.random() < 0.4:
                ebox_pos = random.randint(50, 150)
                ebox = "CACGTG"  # Classic E-box
                for j, base in enumerate(ebox):
                    if ebox_pos + j < seq_length:
                        seq[ebox_pos + j] = base

            sequences.append(''.join(seq))
            labels.append(1)
        else:
            # Generating negative sequence (random DNA)
            seq = random.choices(['A', 'C', 'G', 'T'], k=seq_length)
            sequences.append(''.join(seq))
            labels.append(0)

    # Shuffling to ensure random distribution
    combined = list(zip(sequences, labels))
    random.shuffle(combined)
    sequences, labels = zip(*combined)

    print(f"Generated {len(sequences)} sequences:")
    print(f"  Positive: {sum(labels)} ({sum(labels) / len(labels) * 100:.1f}%)")
    print(f"  Negative: {len(labels) - sum(labels)} ({(len(labels) - sum(labels)) / len(labels) * 100:.1f}%)")

    return list(sequences), list(labels)


def load_data(data_path: str = None):
    """Loading TFBS data (ENCODE)"""
    if data_path is None or not os.path.exists(data_path):
        return generate_patterned_sequences(n_sequences=1000)
    else:
        if load_encode_peaks is not None:
            print(f"\nLoading real ENCODE ChIP-seq data from {data_path}...")
            sequences, labels = load_encode_peaks(data_path, max_sequences=1000)
            if sequences:
                return sequences, labels
        print("\nFalling back to synthetic data.")
        return generate_patterned_sequences(n_sequences=1000)


def evaluate_model(model, dataloader, device):
    """Evaluation of the model"""
    model.eval()

    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids).squeeze()
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).float()

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    metrics = {
        'accuracy': accuracy_score(all_labels, all_predictions),
        'precision': precision_score(all_labels, all_predictions, zero_division=0),
        'recall': recall_score(all_labels, all_predictions, zero_division=0),
        'f1': f1_score(all_labels, all_predictions, zero_division=0),
        'roc_auc': roc_auc_score(all_labels, all_probabilities)
    }

    return metrics


def demonstrate_language_hierarchy(example_sequences):
    """Demonstrating the complete language processing hierarchy"""
    print("\n" + "=" * 80)
    print("FORMAL LANGUAGE HIERARCHY IN DNA ANALYSIS")
    print("=" * 80)

    # LEVEL 1: Regular Expressions
    print(" LEVEL 1: REGULAR EXPRESSIONS ")
    regex_detector = RegexMotifDetector()

    for i, seq in enumerate(example_sequences[:2], 1):
        print(f"\nExample {i}: {seq[:60]}...")
        regex_detector.analyze_sequence(seq, verbose=True)

    # LEVEL 2: Context-Free Grammars
    print(" LEVEL 2: CONTEXT-FREE GRAMMARS ")
    dna_grammar = DNAGrammar()

    grammar_examples = [
        "TATA CAAT TSS",
        "CTCF EBOX CTCF",
        "TATA SPACER CAAT SPACER GC SPACER TSS"
    ]

    for example in grammar_examples:
        print(f"\nTesting structure: {example}")
        is_valid = dna_grammar.validate_structure(example)
        print(f"Valid regulatory structure: {is_valid}")

    # LEVEL 3: Transformers
    print("LEVEL 3: TRANSFORMER NEURAL NETWORKS ")


def compare_approaches(sequences, labels, transformer_model, vocabulary, dna_grammar, device):
    """Comparing all three approaches on the same data"""
    print("\n" + "=" * 80)
    print("COMPARATIVE ANALYSIS: REGEX vs CFG vs TRANSFORMER")
    print("=" * 80)

    regex_detector = RegexMotifDetector()

    results = {
        'regex': {'correct': 0, 'total': 0},
        'cfg': {'correct': 0, 'total': 0},
        'transformer': {'correct': 0, 'total': 0, 'probabilities': []}
    }

    print("\nAnalyzing sample sequences...")

    for i in range(min(20, len(sequences))):
        seq = sequences[i]
        true_label = labels[i]

        print(f"\n{'─' * 60}")
        print(f"Sequence {i + 1}: {seq[:50]}...")
        print(f"True label: {'BINDING' if true_label else 'no binding'}")

        # REGEX
        regex_analysis = regex_detector.analyze_sequence(seq, verbose=False)
        has_ctcf = regex_analysis['has_CTCF']
        promoter_elements = regex_analysis['has_promoter_elements']
        motif_count = regex_analysis['total_motifs_found']

        regex_pred = 1 if (has_ctcf or (promoter_elements and motif_count >= 2)) else 0
        regex_correct = (regex_pred == true_label)
        results['regex']['correct'] += 1 if regex_correct else 0
        results['regex']['total'] += 1

        # CFG
        structure_tokens = []
        if any(m in seq for m in ['TATA', 'TATAAA']):
            structure_tokens.append('TATA')
        if any(m in seq for m in ['CAAT', 'GGTCAATCT']):
            structure_tokens.append('CAAT')
        if 'CTCF' in seq or re.search(r'[AG]CCGCG[CG]', seq):
            structure_tokens.append('CTCF')
        if re.search(r'CA[ATCG]{2}TG', seq):
            structure_tokens.append('EBOX')
        if 'GGGCGG' in seq:
            structure_tokens.append('GC')

        if len(structure_tokens) >= 2:
            if 'TATA' in structure_tokens:
                others = [t for t in structure_tokens if t != 'TATA']
                structure = f"TATA {' SPACER '.join(others)} TSS"
            else:
                structure = ' '.join(structure_tokens[:2])
        else:
            structure = ''

        cfg_valid = dna_grammar.validate_structure(structure) if structure else False
        cfg_pred = 1 if cfg_valid else 0
        cfg_correct = (cfg_pred == true_label)
        results['cfg']['correct'] += 1 if cfg_correct else 0
        results['cfg']['total'] += 1

        # TRANSFORMER
        encoded = vocabulary.encode(seq, max_length=200)
        input_tensor = torch.tensor([encoded], dtype=torch.long).to(device)

        with torch.no_grad():
            transformer_output = transformer_model(input_tensor)
            transformer_prob = torch.sigmoid(transformer_output).item()
            transformer_pred = 1 if transformer_prob > 0.5 else 0
            transformer_correct = (transformer_pred == true_label)

        results['transformer']['correct'] += 1 if transformer_correct else 0
        results['transformer']['total'] += 1
        results['transformer']['probabilities'].append(transformer_prob)

        print(f"  Regex:       {'BINDING' if regex_pred else 'no binding'} ({'✓' if regex_correct else '✗'})")
        print(f"  CFG:         {'BINDING' if cfg_pred else 'no binding'} ({'✓' if cfg_correct else '✗'})")
        print(f"  Transformer: {transformer_prob:.3f} ({'✓' if transformer_correct else '✗'})")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY COMPARISON:")
    print(f"{'=' * 60}")
    print(
        f"Regular Expressions: {results['regex']['correct']}/{results['regex']['total']} = {results['regex']['correct'] / results['regex']['total']:.3f}")
    print(
        f"Context-Free Grammar: {results['cfg']['correct']}/{results['cfg']['total']} = {results['cfg']['correct'] / results['cfg']['total']:.3f}")
    print(
        f"Transformer: {results['transformer']['correct']}/{results['transformer']['total']} = {results['transformer']['correct'] / results['transformer']['total']:.3f}")
    print(f"{'=' * 60}\n")

    return results


def main():
    print("\n" + "=" * 80)
    print("DNA AS FORMAL LANGUAGE: COMPLETE LANGUAGE PROCESSING PIPELINE")
    print("=" * 80)

    # Configuration
    config = {
        'data_path': './ENCFF308JDD.bed',
        'max_seq_length': 200,
        'k': 6,
        'd_model': 128,
        'nhead': 8,
        'num_layers': 4,
        'dropout': 0.1,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'num_epochs': 20,
        'early_stopping_patience': 5,
        'test_split': 0.2,
        'val_split': 0.1,
        'output_dir': './outputs',
        'random_seed': 42,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    os.makedirs(config['output_dir'], exist_ok=True)
    set_random_seed(config['random_seed'])

    # ========== STEP 1: LOADING DATA ==========
    print("\n" + "=" * 80)
    print("STEP 1: LOADING DATA")
    print("=" * 80)

    sequences, labels = load_data(config['data_path'])

    # ========== STEP 2: CREATING VOCABULARY ==========
    print("\n" + "=" * 80)
    print("STEP 2: CREATING DNA VOCABULARY")
    print("=" * 80)
    vocabulary = DNAVocabulary(k=config['k'])

    # ========== STEP 3: DEMONSTRATING LANGUAGE HIERARCHY ==========
    demonstrate_language_hierarchy(sequences[:5])

    # ========== STEP 4: PROPER DATA SPLITTING ==========
    print("\n" + "=" * 80)
    print("STEP 4: SPLITTING DATA")
    print("=" * 80)

    n_total = len(sequences)
    n_test = int(n_total * config['test_split'])
    n_val = int((n_total - n_test) * config['val_split'])
    n_train = n_total - n_test - n_val

    # Shuffling indices randomly
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

    # Verifying distribution
    print(f"\nLabel distribution:")
    print(f"  Train: +{sum(train_labels)}/{len(train_labels)} ({sum(train_labels) / len(train_labels) * 100:.1f}%)")
    print(f"  Val:   +{sum(val_labels)}/{len(val_labels)} ({sum(val_labels) / len(val_labels) * 100:.1f}%)")
    print(f"  Test:  +{sum(test_labels)}/{len(test_labels)} ({sum(test_labels) / len(test_labels) * 100:.1f}%)")

    # ========== STEP 5: CREATING DATASETS ==========
    print("\n" + "=" * 80)
    print("STEP 5: CREATING DATASETS")
    print("=" * 80)

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

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # ========== STEP 6: TRAINING TRANSFORMER ==========
    print("\n" + "=" * 80)
    print("STEP 6: TRAINING TRANSFORMER")
    print("=" * 80)

    transformer_model = TransformerModel(
        vocab_size=vocabulary.vocab_size,
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        max_seq_length=config['max_seq_length']
    )

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
        save_dir=config['output_dir']
    )

    # ========== STEP 7: EVALUATING TRANSFORMER ==========
    print("\n" + "=" * 80)
    print("STEP 7: EVALUATING TRANSFORMER")
    print("=" * 80)

    transformer_metrics = evaluate_model(transformer_model, test_loader, config['device'])
    print(f"\nTransformer Test Metrics:")
    for metric, value in transformer_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # ========== STEP 8: COMPARING ALL APPROACHES ==========
    dna_grammar = DNAGrammar()
    comparison_results = compare_approaches(
        test_sequences,
        test_labels,
        transformer_model,
        vocabulary,
        dna_grammar,
        config['device']
    )

    # ========== STEP 9: ATTENTION VISUALIZATION ==========
    print("\n" + "=" * 80)
    print("STEP 8: ATTENTION VISUALIZATION")
    print("=" * 80)

    visualizer = AttentionVisualizer(transformer_model, vocabulary)

    for i in range(min(3, len(test_sequences))):
        seq = test_sequences[i]
        label = test_labels[i]

        print(f"\nExample {i + 1}:")
        print(f"  Sequence: {seq[:50]}...")
        print(f"  True label: {'Binding' if label else 'No binding'}")

        try:
            attention_data = visualizer.get_attention_weights(seq)
            print(f"  Prediction: {attention_data['prediction']:.3f}")

            heatmap_path = os.path.join(config['output_dir'], f'attention_example_{i + 1}_heatmap.png')
            importance_path = os.path.join(config['output_dir'], f'attention_example_{i + 1}_importance.png')

            visualizer.plot_attention_heatmap(attention_data, save_path=heatmap_path)
            visualizer.plot_sequence_importance(attention_data, save_path=importance_path)
            print(f"  ✓ Visualizations saved")

        except Exception as e:
            print(f"  ⚠️ Visualization error: {e}")


    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved in: {config['output_dir']}")
    print("=" * 80)


if __name__ == "__main__":
    main()