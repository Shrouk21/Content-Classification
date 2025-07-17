import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.append('/content')

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# PyTorch imports
import torch
from collections import Counter
from torch.nn.utils.rnn import pad_sequence

# nltk.download('stopwords', quite=True)
# nltk.download('wordnet', quite=True)

class PyTorchTokenizer:
    """Custom tokenizer for PyTorch"""
    
    def __init__(self, max_vocab_size=10000, oov_token='<OOV>', pad_token='<PAD>'):
        self.max_vocab_size = max_vocab_size
        self.oov_token = oov_token
        self.pad_token = pad_token
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_counts = Counter()
        self.vocab_size = 0
        
    def fit_on_texts(self, texts):
        """Build vocabulary from texts"""
        # Count words
        for text in texts:
            words = text.split()
            self.word_counts.update(words)
        
        # Create vocabulary
        # Reserve indices for special tokens
        self.word_to_idx = {self.pad_token: 0, self.oov_token: 1}
        self.idx_to_word = {0: self.pad_token, 1: self.oov_token}
        
        # Add most common words
        most_common = self.word_counts.most_common(self.max_vocab_size - 2)
        for idx, (word, count) in enumerate(most_common, start=2):
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
            
        self.vocab_size = len(self.word_to_idx)
        
    def texts_to_sequences(self, texts):
        """Convert texts to sequences of indices"""
        sequences = []
        for text in texts:
            words = text.split()
            sequence = [self.word_to_idx.get(word, 1) for word in words]  # 1 is OOV index
            sequences.append(sequence)
        return sequences
    
    def pad_sequences(self, sequences, max_length=100, padding='post'):
        """Pad sequences to fixed length"""
        padded_sequences = []
        
        for seq in sequences:
            if len(seq) > max_length:
                # Truncate
                if padding == 'post':
                    seq = seq[:max_length]
                else:
                    seq = seq[-max_length:]
            else:
                # Pad
                pad_length = max_length - len(seq)
                if padding == 'post':
                    seq = seq + [0] * pad_length
                else:
                    seq = [0] * pad_length + seq
            
            padded_sequences.append(seq)
        
        return np.array(padded_sequences)

class Preprocess:
    def __init__(self, df):
        self.df = df.copy()
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.img_tokenizer = None
        self.query_tokenizer = None
        self.label_encoder = None

    def missing_values(self):
        """
        Check and handle missing values
        """
        print(f"Missing values before cleaning: {self.df.isnull().sum()}")
        
        # Fill missing values or drop rows
        self.df = self.df.dropna()
        
        print(f"Missing values after cleaning: {self.df.isnull().sum()}")
        return self.df

    def clean_text(self, text):
        """
        Clean text data for LSTM input
        """
        if pd.isna(text):
            return ""
        
        # Convert to string
        text = str(text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Tokenization
        tokens = text.split()
        
        # Remove stop words and lemmatize
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        
        return ' '.join(tokens)

    def create_sequences(self, text_data, max_vocab_size=10000, max_sequence_length=100):
        """
        Convert text to sequences for LSTM input using PyTorch
        """
        tokenizer = PyTorchTokenizer(max_vocab_size=max_vocab_size)
        tokenizer.fit_on_texts(text_data)
        sequences = tokenizer.texts_to_sequences(text_data)
        padded_sequences = tokenizer.pad_sequences(sequences, max_length=max_sequence_length)
        
        return padded_sequences, tokenizer

    def prepare_multi_input_data(self, img_desc_col='image_descriptions', query_col='query', 
                                max_vocab_size=10000, max_seq_length=100):
        """
        Prepare separate inputs for image descriptions and queries
        """
        print("Cleaning text data...")
        
        # Clean both text inputs
        self.df['cleaned_image_desc'] = self.df[img_desc_col].apply(self.clean_text)
        self.df['cleaned_query'] = self.df[query_col].apply(self.clean_text)
        
        print("Creating sequences...")
        
        # Create sequences for both inputs
        img_sequences, self.img_tokenizer = self.create_sequences(
            self.df['cleaned_image_desc'], max_vocab_size, max_seq_length
        )
        query_sequences, self.query_tokenizer = self.create_sequences(
            self.df['cleaned_query'], max_vocab_size, max_seq_length
        )
        
        print(f"Image sequences shape: {img_sequences.shape}")
        print(f"Query sequences shape: {query_sequences.shape}")
        
        return img_sequences, query_sequences

    def encode_labels(self, target_column='toxic_category'):
        """
        Encode target labels for classification
        """
        print(f"Encoding labels from column: {target_column}")
        print(f"Unique classes: {self.df[target_column].unique()}")
        
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(self.df[target_column])
        
        # For multi-class classification with PyTorch
        num_classes = len(self.label_encoder.classes_)
        
        if num_classes == 2:
            # Binary classification - use float labels
            categorical_labels = encoded_labels.astype(np.float32)
        else:
            # Multi-class classification - use one-hot encoding
            categorical_labels = np.eye(num_classes)[encoded_labels].astype(np.float32)
        
        print(f"Number of classes: {num_classes}")
        print(f"Label shape: {categorical_labels.shape}")
        
        return categorical_labels, self.label_encoder

    def split_data(self, img_sequences, query_sequences, labels, test_size=0.2, val_size=0.1):
        """
        Split data for training, validation, and testing
        """
        print("Splitting data...")
        
        # Handle stratification based on label format
        if labels.ndim == 1:
            # Binary classification
            stratify_labels = labels
        else:
            # Multi-class classification - use argmax for stratification
            stratify_labels = np.argmax(labels, axis=1)
        
        # First split: train+val vs test
        img_train_val, img_test, query_train_val, query_test, y_train_val, y_test = train_test_split(
            img_sequences, query_sequences, labels, 
            test_size=test_size, random_state=42, stratify=stratify_labels
        )
        
        # Update stratify labels for second split
        if y_train_val.ndim == 1:
            stratify_train_val = y_train_val
        else:
            stratify_train_val = np.argmax(y_train_val, axis=1)
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        img_train, img_val, query_train, query_val, y_train, y_val = train_test_split(
            img_train_val, query_train_val, y_train_val,
            test_size=val_size_adjusted, random_state=42, stratify=stratify_train_val
        )
        
        print(f"Train set: {img_train.shape[0]} samples")
        print(f"Validation set: {img_val.shape[0]} samples")
        print(f"Test set: {img_test.shape[0]} samples")
        
        return {
            'train': (img_train, query_train, y_train),
            'val': (img_val, query_val, y_val),
            'test': (img_test, query_test, y_test)
        }

    def get_vocab_info(self):
        """
        Get vocabulary information for model building
        """
        img_vocab_size = self.img_tokenizer.vocab_size if self.img_tokenizer else 0
        query_vocab_size = self.query_tokenizer.vocab_size if self.query_tokenizer else 0
        
        return {
            'img_vocab_size': img_vocab_size,
            'query_vocab_size': query_vocab_size,
            'num_classes': len(self.label_encoder.classes_) if self.label_encoder else 0
        }

    def full_preprocessing_pipeline(self, img_desc_col='image_descriptions', 
                                   query_col='query', target_col='toxic_category',
                                   max_vocab_size=10000, max_seq_length=100):
        """
        Complete preprocessing pipeline
        """
        print("Starting full preprocessing pipeline...")
        
        # Handle missing values
        self.missing_values()
        
        # Prepare multi-input data
        img_sequences, query_sequences = self.prepare_multi_input_data(
            img_desc_col, query_col, max_vocab_size, max_seq_length
        )
        
        # Encode labels
        labels, _ = self.encode_labels(target_col)
        
        # Split data
        data_splits = self.split_data(img_sequences, query_sequences, labels)
        
        # Get vocabulary info
        vocab_info = self.get_vocab_info()
        
        print("Preprocessing complete!")
        print(f"Vocabulary info: {vocab_info}")
        
        return data_splits, vocab_info

    def save_tokenizers(self, filepath_prefix='tokenizers'):
        """
        Save tokenizers for future use
        """
        import pickle
        
        if self.img_tokenizer:
            with open(f'{filepath_prefix}_img_tokenizer.pkl', 'wb') as f:
                pickle.dump(self.img_tokenizer, f)
        
        if self.query_tokenizer:
            with open(f'{filepath_prefix}_query_tokenizer.pkl', 'wb') as f:
                pickle.dump(self.query_tokenizer, f)
        
        if self.label_encoder:
            with open(f'{filepath_prefix}_label_encoder.pkl', 'wb') as f:
                pickle.dump(self.label_encoder, f)
        
        print(f"Tokenizers saved with prefix: {filepath_prefix}")

    def load_tokenizers(self, filepath_prefix='tokenizers'):
        """
        Load saved tokenizers
        """
        import pickle
        
        try:
            with open(f'{filepath_prefix}_img_tokenizer.pkl', 'rb') as f:
                self.img_tokenizer = pickle.load(f)
            
            with open(f'{filepath_prefix}_query_tokenizer.pkl', 'rb') as f:
                self.query_tokenizer = pickle.load(f)
            
            with open(f'{filepath_prefix}_label_encoder.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            print(f"Tokenizers loaded from prefix: {filepath_prefix}")
        except FileNotFoundError as e:
            print(f"Could not load tokenizers: {e}")

    def preprocess_new_data(self, img_desc_text, query_text, max_seq_length=100):
        """
        Preprocess new data using existing tokenizers
        """
        if not self.img_tokenizer or not self.query_tokenizer:
            raise ValueError("Tokenizers not found. Please run preprocessing pipeline first or load tokenizers.")
        
        # Clean texts
        cleaned_img_desc = self.clean_text(img_desc_text)
        cleaned_query = self.clean_text(query_text)
        
        # Convert to sequences
        img_seq = self.img_tokenizer.texts_to_sequences([cleaned_img_desc])
        query_seq = self.query_tokenizer.texts_to_sequences([cleaned_query])
        
        # Pad sequences
        img_padded = self.img_tokenizer.pad_sequences(img_seq, max_length=max_seq_length)
        query_padded = self.query_tokenizer.pad_sequences(query_seq, max_length=max_seq_length)
        
        return img_padded[0], query_padded[0]

