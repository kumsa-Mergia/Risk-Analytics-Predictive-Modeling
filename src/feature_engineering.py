import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()

    def clean_data(self):
        # Example cleaning logic:
        # Drop columns with too many missing values
        threshold = 0.9 * len(self.df)
        self.df.dropna(thresh=threshold, axis=1, inplace=True)

        # Fill some missing values as example
        if 'CustomValueEstimate' in self.df.columns:
            self.df['CustomValueEstimate'] = self.df['CustomValueEstimate'].fillna(self.df['CustomValueEstimate'].median())

        for col in ['Gender', 'Bank', 'AccountType', 'MaritalStatus']:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna('Unknown')

    def encode_categoricals(self):
        # Example encoding categorical variables
        categorical_cols = self.df.select_dtypes(include='object').columns.tolist()
        self.df = pd.get_dummies(self.df, columns=categorical_cols, drop_first=True)

    def create_features(self):
        # Example feature engineering
        if 'TotalClaims' in self.df.columns and 'TotalPremium' in self.df.columns:
            self.df['LossRatio'] = self.df['TotalClaims'] / self.df['TotalPremium']
            self.df['Margin'] = self.df['TotalPremium'] - self.df['TotalClaims']

    def get_data(self):
        return self.df

    def get_features_and_target(self, target):
        """
        Returns features (X) and target (y) DataFrames/Series
        Drops the target column from features.
        """
        if target not in self.df.columns:
            raise ValueError(f"Target column '{target}' not found in dataframe")

        X = self.df.drop(columns=[target])
        y = self.df[target]
        return X, y
