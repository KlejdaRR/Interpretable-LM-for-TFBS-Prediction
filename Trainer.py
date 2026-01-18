"""
This module handles training the model.

- Training loop: Repeatedly showing data to model, calculate error, update weights
- Validation: Checking performance on unseen data to detect overfitting
- Loss function: Measuring how wrong the model's predictions are
- Optimizer: Algorithm that updates model weights to reduce loss
- Early stopping: Stopping training when performance stops improving
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List
import time
import os


class Trainer:
    """
    A trainer class that handles the training process.

    During training
    1. Show model some data (forward pass)
    2. Calculate how wrong it was (compute loss)
    3. Update model weights to do better (backward pass)
    4. Repeat many times
    """

    def __init__(self,
                 model: nn.Module,
                 device: str = None,
                 learning_rate: float = 1e-4):
        """
        Initializing the trainer.

        It will take as arguments:
            model: The neural network to train
            device: 'cuda' for GPU or 'cpu' for CPU
            learning_rate: How big the update steps should be
                          - Too large: training unstable, might diverge
                          - Too small: training very slow
                          - 1e-4 (0.0001) is a good default
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"\nTrainer initialized:")
        print(f"  - Device: {self.device}")
        print(f"  - Learning rate: {learning_rate}")

        self.model = model.to(self.device)

        # OPTIMIZER: How to update model weights
        # We chose to use Adam because it is a popular optimizer that works well for most problems
        # It automatically adjusts learning rate for each parameter
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5  # L2 regularization (in order to prevent overfitting)
        )

        # LEARNING RATE SCHEDULER: Adjusting learning rate during training
        # Reducing learning rate when validation loss stops improving
        # This helps fine-tune the model in later stages
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',  # Minimize validation loss
            factor=0.5,  # Reduce LR by half
            patience=3  # Wait 3 epochs before reducing

        )

        # LOSS FUNCTION: Measuring the error
        # BCEWithLogitsLoss = Binary Cross Entropy Loss
        self.criterion = nn.BCEWithLogitsLoss()

        # Training history (for plotting later)
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

    def train_one_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Method that trains for one epoch (one pass through all training data).

        It will take as arguments:
            dataloader: DataLoader that provides batches of data

        It will return:
            Dictionary with average loss and accuracy
        """
        # Setting model to training mode
        # This enables dropout and batch normalization training behavior
        self.model.train()

        # Track statistics
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        # Iterate through batches
        for batch_idx, batch in enumerate(dataloader):
            # STEP 1: Move data to device (GPU/CPU)
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['label'].to(self.device)

            # STEP 2: Forward pass - get model predictions
            predictions = self.model(input_ids).squeeze()

            # STEP 3: Calculate loss (how wrong are we?)
            loss = self.criterion(predictions, labels)

            # STEP 4: Backward pass - calculate gradients
            # Reset gradients from previous iteration
            self.optimizer.zero_grad()

            # Calculate gradients (how to change weights to reduce loss)
            loss.backward()

            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # STEP 5: Update model weights
            self.optimizer.step()

            # STEP 6: Track statistics
            total_loss += loss.item()

            # Convert predictions to binary (0 or 1)
            # Sigmoid converts raw scores to probabilities [0, 1]
            # Then threshold at 0.5
            binary_preds = (torch.sigmoid(predictions) > 0.5).float()
            correct_predictions += (binary_preds == labels).sum().item()
            total_samples += labels.size(0)

            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(dataloader)}, "
                      f"Loss: {loss.item():.4f}")

        # Calculate averages
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_samples

        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Method that evaluates the model on validation data.

        This tells us how well the model generalizes to new data.

        It will take as arguments:
            dataloader: DataLoader with validation data

        It will return:
            Dictionary with average loss and accuracy
        """
        # Set model to evaluation mode
        # This disables dropout and sets batch normalization to eval mode
        self.model.eval()

        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        # Don't calculate gradients (saves memory and computation)
        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['label'].to(self.device)

                # Forward pass
                predictions = self.model(input_ids).squeeze()

                # Calculate loss
                loss = self.criterion(predictions, labels)

                # Track statistics
                total_loss += loss.item()
                binary_preds = (torch.sigmoid(predictions) > 0.5).float()
                correct_predictions += (binary_preds == labels).sum().item()
                total_samples += labels.size(0)

        # Calculate averages
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_samples

        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }

    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int = 20,
              early_stopping_patience: int = 5,
              save_dir: str = '.') -> Dict[str, List]:
        """
        Method that completes training loop with early stopping.

        It will take as arguments:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            num_epochs: Maximum number of epochs to train
            early_stopping_patience: Stop if no improvement for this many epochs
            save_dir: Directory to save the best model (default: current directory)

        It will return:
            Training history dictionary
        """
        print(f"\n{'=' * 70}")
        print("STARTING TRAINING")
        print(f"{'=' * 70}")
        print(f"Training for up to {num_epochs} epochs")
        print(f"Early stopping patience: {early_stopping_patience}")

        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        model_save_path = os.path.join(save_dir, 'best_model.pt')
        print(f"Model will be saved to: {model_save_path}")

        best_val_loss = float('inf')
        patience_counter = 0
        start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start = time.time()

            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 70)

            # TRAINING PHASE
            print("Training...")
            train_metrics = self.train_one_epoch(train_loader)

            # VALIDATION PHASE
            print("Validating...")
            val_metrics = self.validate(val_loader)

            # UPDATE LEARNING RATE
            self.scheduler.step(val_metrics['loss'])

            # SAVE METRICS
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])

            # PRINT PROGRESS
            epoch_time = time.time() - epoch_start
            print(f"\nResults:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}, "
                  f"Train Acc: {train_metrics['accuracy']:.4f}")
            print(f"  Val Loss:   {val_metrics['loss']:.4f}, "
                  f"Val Acc:   {val_metrics['accuracy']:.4f}")
            print(f"  Time: {epoch_time:.2f}s")

            # EARLY STOPPING CHECK
            if val_metrics['loss'] < best_val_loss:
                # Improvement! Save model and reset patience
                best_val_loss = val_metrics['loss']
                patience_counter = 0

                # Save best model
                try:
                    torch.save(self.model.state_dict(), model_save_path)
                    print(f"  ✓ New best model saved! (Val Loss: {best_val_loss:.4f})")
                except Exception as e:
                    print(f"  ⚠ Warning: Could not save model: {e}")
            else:
                # No improvement
                patience_counter += 1
                print(f"  No improvement. Patience: {patience_counter}/{early_stopping_patience}")

                if patience_counter >= early_stopping_patience:
                    print(f"\n{'=' * 70}")
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    print(f"{'=' * 70}")
                    break

        # TRAINING COMPLETE
        total_time = time.time() - start_time
        print(f"\n{'=' * 70}")
        print("TRAINING COMPLETE")
        print(f"{'=' * 70}")
        print(f"Total time: {total_time:.2f}s ({total_time / 60:.2f} minutes)")
        print(f"Best validation loss: {best_val_loss:.4f}")

        # Load best model
        if os.path.exists(model_save_path):
            print(f"\nLoading best model from {model_save_path}...")
            try:
                self.model.load_state_dict(torch.load(model_save_path))
                print("Best model loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load best model: {e}")
        else:
            print(f"\nWarning: Best model file not found at {model_save_path}")

        return self.history

