import torch
import torch.nn as nn

class TFBSTrainer:
    """
    Training class with advanced ML techniques.

    Demonstrates AML concepts:
    - Adam optimization
    - Learning rate scheduling
    - Gradient clipping
    - Early stopping
    """

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device

        # Optimizer with weight decay (L2 regularization)
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-4,
            weight_decay=1e-5
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )

        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

    def train_epoch(self, dataloader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in dataloader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(input_ids)
            loss = self.criterion(outputs.squeeze(), labels)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            predictions = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total

        return avg_loss, accuracy

    def validate(self, dataloader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids)
                loss = self.criterion(outputs.squeeze(), labels)

                total_loss += loss.item()
                predictions = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total

        return avg_loss, accuracy

    def train(self, train_loader, val_loader, num_epochs: int = 50,
              early_stopping_patience: int = 5):
        """
        Complete training loop with early stopping.
        """
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validation
            val_loss, val_acc = self.validate(val_loader)

            # Update learning rate
            self.scheduler.step(val_loss)

            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)

            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print("-" * 50)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), '/home/claude/best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break

        # Load best model
        self.model.load_state_dict(torch.load('/home/claude/best_model.pt'))