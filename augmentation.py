# augmentor.py

import pandas as pd
import numpy as np
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
import nlpaug.augmenter.word as naw
from collections import Counter

class DataAugmentor:
    def __init__(self, df, text_cols=['query', 'image_descriptions'], label_col='toxic_category'):
        self.df = df.copy()
        self.text_cols = text_cols
        self.label_col = label_col
        self.augmented_df = None

    def show_class_distribution(self):
        counts = self.df[self.label_col].value_counts()
        print("Class Distribution:\n", counts)

    def augment_text(self, method='synonym', num_augments_per_sample=1):
        """
        Applies synonym-based augmentation on minority classes.
        """
        augmenter = naw.SynonymAug(aug_src='wordnet')

        # Get class distribution
        class_counts = self.df[self.label_col].value_counts()
        max_count = class_counts.max()

        augmented_data = []
        for cls in class_counts.index:
            df_cls = self.df[self.df[self.label_col] == cls]
            n_needed = max_count - len(df_cls)

            if n_needed <= 0:
                continue

            sampled = resample(df_cls, n_samples=n_needed, replace=True, random_state=42)

            for _, row in sampled.iterrows():
                augmented_row = row.copy()
                for col in self.text_cols:
                    text = row[col]
                    aug_text = augmenter.augment(text, n=num_augments_per_sample)
                    augmented_row[col] = aug_text
                augmented_data.append(augmented_row)

        augmented_df = pd.DataFrame(augmented_data)
        self.augmented_df = pd.concat([self.df, augmented_df], ignore_index=True)
        return self.augmented_df

    def oversample(self):
        """
        Basic oversampling of minority classes without text modification.
        """
        max_size = self.df[self.label_col].value_counts().max()
        lst = [self.df]
        for class_index, group in self.df.groupby(self.label_col):
            samples_needed = max_size - len(group)
            if samples_needed > 0:
                lst.append(group.sample(samples_needed, replace=True, random_state=42))
        self.augmented_df = pd.concat(lst).sample(frac=1, random_state=42).reset_index(drop=True)
        return self.augmented_df

    def apply_smote(self, vectorized_features, labels):
        """
        Apply SMOTE to vectorized (e.g., TF-IDF) features.
        NOTE: Only use if you already have vectorized data.
        """
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(vectorized_features, labels)
        return X_res, y_res

    def save_augmented(self, path="augmented_train.csv"):
        if self.augmented_df is not None:
            self.augmented_df.to_csv(path, index=False)
            print(f"Augmented training data saved to: {path}")
        else:
            print("No augmented data to save.")
