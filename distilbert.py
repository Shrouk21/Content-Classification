import torch, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiInputDataset(Dataset):
    def __init__(self, a, b, y, tokenizer, max_len):
        self.encodings = tokenizer(list(a), list(b), truncation=True, padding='max_length', max_length=max_len)
        self.labels = torch.tensor(y)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k,v in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

class DistilBertLoRATrainer:
    def __init__(self, model_name="distilbert-base-uncased", num_labels=2, max_len=128):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        lora_config = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.1, bias="none",
                                 target_modules=["q_lin", "v_lin"], task_type=TaskType.SEQ_CLS)
        self.model = get_peft_model(base_model, lora_config).to(device)
        self.max_len = max_len
        self.train_losses, self.val_losses = [], []
        self.train_accs, self.val_accs = [], []

    def prepare_data(self, df, col_a, col_b, label_col, test_size=0.2, val_size=0.1):
        le = LabelEncoder()
        y = le.fit_transform(df[label_col])
        a, b = df[col_a].astype(str).values, df[col_b].astype(str).values

        a_trainval, a_test, b_trainval, b_test, y_trainval, y_test = train_test_split(
            a, b, y, test_size=test_size, stratify=y, random_state=42
        )
        val_ratio = val_size / (1 - test_size)
        a_train, a_val, b_train, b_val, y_train, y_val = train_test_split(
            a_trainval, b_trainval, y_trainval, test_size=val_ratio, stratify=y_trainval, random_state=42
        )

        self.train_data = MultiInputDataset(a_train, b_train, y_train, self.tokenizer, self.max_len)
        self.val_data = MultiInputDataset(a_val, b_val, y_val, self.tokenizer, self.max_len)
        self.test_data = MultiInputDataset(a_test, b_test, y_test, self.tokenizer, self.max_len)
        self.label_encoder = le
        self.y_test = y_test

    def train(self, batch_size=16, lr=2e-5, epochs=4):
        train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_data, batch_size=batch_size)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        best_val_acc = 0
        for epoch in range(epochs):
            self.model.train()
            total_loss, total_correct = 0, 0
            for batch in train_loader:
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(device)
                optimizer.zero_grad()
                outputs = self.model(**inputs)
                loss = F.cross_entropy(outputs.logits, labels)
                loss.backward()
                optimizer.step()
                preds = outputs.logits.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total_loss += loss.item()

            train_loss = total_loss / len(train_loader)
            train_acc = total_correct / len(self.train_data)
            val_loss, val_acc = self.evaluate(val_loader, mode="val")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), "best_model.pt")

            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | Val Loss={val_loss:.4f}, Acc={val_acc:.4f}")

    def evaluate(self, loader, mode="test"):
        self.model.eval()
        total_loss, total_correct = 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in loader:
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(device)
                outputs = self.model(**inputs)
                loss = F.cross_entropy(outputs.logits, labels)
                preds = outputs.logits.argmax(dim=1)
                total_loss += loss.item()
                total_correct += (preds == labels).sum().item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(loader)
        accuracy = total_correct / len(loader.dataset)

        if mode == "test":
            print(f"\nTest Accuracy: {accuracy:.4f}")
            report = classification_report(all_labels, all_preds, target_names=self.label_encoder.classes_)
            cm = confusion_matrix(all_labels, all_preds)
            print(report)
            return accuracy, report, cm, all_preds
        return avg_loss, accuracy

    def test(self, batch_size=16):
        loader = DataLoader(self.test_data, batch_size=batch_size)
        return self.evaluate(loader, mode="test")

    def plot_metrics(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(self.train_losses, label="Train Loss")
        ax1.plot(self.val_losses, label="Val Loss")
        ax1.set_title("Loss"); ax1.legend(); ax1.grid(True)

        ax2.plot(self.train_accs, label="Train Acc")
        ax2.plot(self.val_accs, label="Val Acc")
        ax2.set_title("Accuracy"); ax2.legend(); ax2.grid(True)
        plt.tight_layout(); plt.show()

    def plot_confusion_matrix(self, cm):
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
        plt.show()


