import pandas as pd
import numpy as np
from sklearn.utils import resample
import nlpaug.augmenter.word as naw

class DataAugmentor:
    def __init__(self, text_cols=['query', 'image descriptions'], label_col='Toxic Category'):
        self.text_cols = text_cols
        self.label_col = label_col

    def augment_df(self, df, method='synonym', val_size=0.1, test_size=0.2):
        from sklearn.model_selection import train_test_split

        # Split into train/val/test using same logic as trainer
        y = df[self.label_col]
        train_val, test = train_test_split(df, test_size=test_size, stratify=y, random_state=42)
        train, val = train_test_split(train_val, test_size=val_size / (1 - test_size), stratify=train_val[self.label_col], random_state=42)

        if method == 'synonym':
            train = self._synonym_augment(train)
        elif method == 'oversample':
            train = self._oversample(train)
        else:
            raise ValueError(f"Unsupported method: {method}")

        # Recombine
        final_df = pd.concat([train, val, test], ignore_index=True)
        return final_df

    def _synonym_augment(self, df):
        augmenter = naw.SynonymAug(aug_src='wordnet')
        class_counts = df[self.label_col].value_counts()
        max_count = class_counts.max()
        augmented_data = []

        for cls in class_counts.index:
            subset = df[df[self.label_col] == cls]
            n_needed = max_count - len(subset)
            if n_needed <= 0:
                continue
            samples = resample(subset, n_samples=n_needed, replace=True, random_state=42)
            for _, row in samples.iterrows():
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
        all_dfs = [df]
        for cls, group in df.groupby(self.label_col):
            needed = max_size - len(group)
            if needed > 0:
                all_dfs.append(group.sample(needed, replace=True, random_state=42))
        return pd.concat(all_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
