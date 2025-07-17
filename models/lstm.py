import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
from tqdm import tqdm
import os

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class MultiInputDataset(Dataset):
    """Custom Dataset for multi-input LSTM"""
    
    def __init__(self, img_sequences, query_sequences, labels):
        self.img_sequences = torch.LongTensor(img_sequences)
        self.query_sequences = torch.LongTensor(query_sequences)
        self.labels = torch.FloatTensor(labels)
        
    def __len__(self):
        return len(self.img_sequences)
    
    def __getitem__(self, idx):
        return {
            'img_seq': self.img_sequences[idx],
            'query_seq': self.query_sequences[idx],
            'label': self.labels[idx]
        }

class MultiInputLSTM(nn.Module):
    """Multi-Input LSTM Classifier"""
    
    def __init__(self, img_vocab_size, query_vocab_size, num_classes, 
                 embedding_dim=128, lstm_hidden_size=64, num_layers=2, 
                 dropout_rate=0.3, bidirectional=True, architecture='basic'):
        super(MultiInputLSTM, self).__init__()
        
        self.img_vocab_size = img_vocab_size
        self.query_vocab_size = query_vocab_size
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        self.architecture = architecture
        
        # Embedding layers
        self.img_embedding = nn.Embedding(img_vocab_size, embedding_dim, padding_idx=0)
        self.query_embedding = nn.Embedding(query_vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM layers
        self.img_lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        self.query_lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate LSTM output size
        lstm_output_size = lstm_hidden_size * 2 if bidirectional else lstm_hidden_size
        
        # Attention mechanism (if using attention architecture)
        if architecture == 'attention':
            self.img_attention = nn.Linear(lstm_output_size, 1)
            self.query_attention = nn.Linear(lstm_output_size, 1)
        
        # Classification head
        combined_size = lstm_output_size * 2  # Two LSTM outputs combined
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) > 1:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def attention_pooling(self, lstm_output, attention_layer):
        """Apply attention pooling to LSTM output"""
        # lstm_output shape: (batch_size, seq_len, hidden_size)
        attention_weights = torch.softmax(attention_layer(lstm_output), dim=1)
        weighted_output = torch.sum(attention_weights * lstm_output, dim=1)
        return weighted_output
    
    def forward(self, img_seq, query_seq):
        # Embedding
        img_embedded = self.img_embedding(img_seq)  # (batch_size, seq_len, embedding_dim)
        query_embedded = self.query_embedding(query_seq)
        
        # LSTM processing
        img_lstm_out, (img_hidden, _) = self.img_lstm(img_embedded)
        query_lstm_out, (query_hidden, _) = self.query_lstm(query_embedded)
        
        # Feature extraction based on architecture
        if self.architecture == 'attention':
            img_features = self.attention_pooling(img_lstm_out, self.img_attention)
            query_features = self.attention_pooling(query_lstm_out, self.query_attention)
        else:
            # Use last hidden state (or mean of bidirectional)
            if self.bidirectional:
                img_features = torch.cat([img_hidden[-2], img_hidden[-1]], dim=1)
                query_features = torch.cat([query_hidden[-2], query_hidden[-1]], dim=1)
            else:
                img_features = img_hidden[-1]
                query_features = query_hidden[-1]
        
        # Combine features
        combined_features = torch.cat([img_features, query_features], dim=1)
        
        # Classification
        output = self.classifier(combined_features)
        
        return output

class LSTMTrainer:
    """LSTM Model Training and Evaluation"""
    
    def __init__(self, model, device=device):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_epoch(self, dataloader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            img_seq = batch['img_seq'].to(self.device)
            query_seq = batch['query_seq'].to(self.device)
            labels = batch['label'].to(self.device)
            
            optimizer.zero_grad()
            
            outputs = self.model(img_seq, query_seq)
            
            if self.model.num_classes == 2:
                # Binary classification
                loss = criterion(outputs.squeeze(), labels.squeeze())
                predictions = torch.sigmoid(outputs.squeeze()) > 0.5
                correct = (predictions == labels.squeeze()).sum().item()
            else:
                # Multi-class classification
                loss = criterion(outputs, labels.argmax(dim=1))
                predictions = torch.argmax(outputs, dim=1)
                correct = (predictions == labels.argmax(dim=1)).sum().item()
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_correct += correct
            total_samples += labels.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def validate_epoch(self, dataloader, criterion):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                img_seq = batch['img_seq'].to(self.device)
                query_seq = batch['query_seq'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(img_seq, query_seq)
                
                if self.model.num_classes == 2:
                    # Binary classification
                    loss = criterion(outputs.squeeze(), labels.squeeze())
                    predictions = torch.sigmoid(outputs.squeeze()) > 0.5
                    correct = (predictions == labels.squeeze()).sum().item()
                else:
                    # Multi-class classification
                    loss = criterion(outputs, labels.argmax(dim=1))
                    predictions = torch.argmax(outputs, dim=1)
                    correct = (predictions == labels.argmax(dim=1)).sum().item()
                
                total_loss += loss.item()
                total_correct += correct
                total_samples += labels.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, num_epochs=50, learning_rate=0.001, 
              weight_decay=1e-5, patience=10, save_path='best_model.pth'):
        """Full training loop"""
        
        # Setup optimizer and criterion
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        if self.model.num_classes == 2:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validation
            val_loss, val_acc = self.validate_epoch(val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print("-" * 50)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), save_path)
                print(f"New best model saved with validation loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load(save_path))
        print("Training completed!")
    
    def evaluate(self, test_loader, label_encoder=None):
        """Evaluate model on test set"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                img_seq = batch['img_seq'].to(self.device)
                query_seq = batch['query_seq'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(img_seq, query_seq)
                
                if self.model.num_classes == 2:
                    predictions = torch.sigmoid(outputs.squeeze()) > 0.5
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.squeeze().cpu().numpy())
                else:
                    predictions = torch.argmax(outputs, dim=1)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.argmax(dim=1).cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        
        if label_encoder:
            target_names = label_encoder.classes_
        else:
            target_names = None
        
        report = classification_report(all_labels, all_predictions, target_names=target_names)
        cm = confusion_matrix(all_labels, all_predictions)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        
        return accuracy, report, cm, all_predictions, all_labels
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, cm, label_encoder=None):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        
        if label_encoder:
            labels = label_encoder.classes_
        else:
            labels = None
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

# Example usage function
def build_and_train_model(data_splits, vocab_info, batch_size=32, embedding_dim=128, 
                         lstm_hidden_size=64, num_layers=2, dropout_rate=0.3, 
                         bidirectional=True, architecture='basic', num_epochs=50,
                         learning_rate=0.001, label_encoder=None):
    """
    Complete model building and training pipeline
    """
    
    # Create datasets
    train_dataset = MultiInputDataset(*data_splits['train'])
    val_dataset = MultiInputDataset(*data_splits['val'])
    test_dataset = MultiInputDataset(*data_splits['test'])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Build model
    model = MultiInputLSTM(
        img_vocab_size=vocab_info['img_vocab_size'],
        query_vocab_size=vocab_info['query_vocab_size'],
        num_classes=vocab_info['num_classes'],
        embedding_dim=embedding_dim,
        lstm_hidden_size=lstm_hidden_size,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        bidirectional=bidirectional,
        architecture=architecture
    )
    
    print(f"Model built with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    trainer = LSTMTrainer(model)
    
    # Train model
    trainer.train(train_loader, val_loader, num_epochs=num_epochs, 
                 learning_rate=learning_rate)
    
    # Evaluate model
    accuracy, report, cm, predictions, labels = trainer.evaluate(test_loader, label_encoder)
    
    # Plot results
    trainer.plot_training_history()
    trainer.plot_confusion_matrix(cm, label_encoder)
    
    return trainer, model, accuracy, report

# Example usage:
"""
# Assuming you have your preprocessed data from the previous script
# data_splits, vocab_info = preprocessor.full_preprocessing_pipeline(...)

trainer, model, accuracy, report = build_and_train_model(
    data_splits=data_splits,
    vocab_info=vocab_info,
    batch_size=32,
    embedding_dim=128,
    lstm_hidden_size=64,
    num_layers=2,
    dropout_rate=0.3,
    bidirectional=True,
    architecture='attention',  # or 'basic'
    num_epochs=50,
    learning_rate=0.001,
    label_encoder=preprocessor.label_encoder
)
"""