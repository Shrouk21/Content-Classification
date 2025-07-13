import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import nlpaug.augmenter.word as naw
from collections import Counter

class DataAugmentor:
    def __init__(self, df, text_cols=['query', 'image_descriptions'], label_col='toxic_category'):
        self.df = df.copy()
        self.text_cols = text_cols
        self.label_col = label_col

    def show_class_distribution(self, df=None):
        df = df if df is not None else self.df
        counts = df[self.label_col].value_counts()
        print("Class Distribution:\n", counts)

    def _augment_text(self, df, num_augments_per_sample=1):
        augmenter = naw.SynonymAug(aug_src='wordnet')
        class_counts = df[self.label_col].value_counts()
        max_count = class_counts.max()
        augmented_data = []

        for cls in class_counts.index:
            df_cls = df[df[self.label_col] == cls]
            n_needed = max_count - len(df_cls)
            if n_needed <= 0:
                continue
            sampled = resample(df_cls, n_samples=n_needed, replace=True, random_state=42)

            for _, row in sampled.iterrows():
                new_row = row.copy()
                for col in self.text_cols:
                    try:
                        new_row[col] = augmenter.augment(row[col])[0]
                    except:
                        new_row[col] = row[col]
                augmented_data.append(new_row)

        augmented_df = pd.DataFrame(augmented_data)
        return pd.concat([df, augmented_df], ignore_index=True)

    def _oversample(self, df):
        max_size = df[self.label_col].value_counts().max()
        lst = [df]
        for _, group in df.groupby(self.label_col):
            samples_needed = max_size - len(group)
            if samples_needed > 0:
                lst.append(group.sample(samples_needed, replace=True, random_state=42))
        return pd.concat(lst).sample(frac=1, random_state=42).reset_index(drop=True)

    def augment_split(self, method='synonym', test_size=0.2, val_size=0.1):
        df = self.df
        y = df[self.label_col]

        # Split: train+val vs test
        train_val_df, test_df = train_test_split(df, test_size=test_size, stratify=y, random_state=42)

        # Adjust val_size to be a fraction of train+val
        val_size_adjusted = val_size / (1 - test_size)
        y_train_val = train_val_df[self.label_col]
        train_df, val_df = train_test_split(train_val_df, test_size=val_size_adjusted, stratify=y_train_val, random_state=42)

        print("Original Train Set Distribution:")
        self.show_class_distribution(train_df)

        if method == 'synonym':
            train_df = self._augment_text(train_df)
        elif method == 'oversample':
            train_df = self._oversample(train_df)

        print("Augmented Train Set Distribution:")
        self.show_class_distribution(train_df)

        return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
