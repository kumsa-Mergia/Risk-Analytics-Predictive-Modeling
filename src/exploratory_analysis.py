import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class InsuranceEDA:
    def __init__(self, filepath, delimiter='|'):
        self.filepath = filepath
        self.delimiter = delimiter
        self.df = pd.read_csv(self.filepath, delimiter=self.delimiter, low_memory=False)
        self._convert_dates()

    def load_data(self):
        self.df = pd.read_csv(self.filepath, delimiter=self.delimiter, low_memory=False)
        self._convert_dates()

    def _convert_dates(self):
        # List of formats to try if default parsing fails
        alternative_formats = [
            '%Y-%m-%d',           # e.g. 2023-04-15
            '%d/%m/%Y',           # e.g. 15/04/2023
            '%m/%d/%Y',           # e.g. 04/15/2023
            '%Y-%m-%d %H:%M:%S',  # e.g. 2023-04-15 13:45:30
            # add other relevant formats as needed
        ]

        for col in ['TransactionMonth', 'VehicleIntroDate']:
            if col in self.df.columns:
                # First try default parsing (dateutil)
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')

                failed_mask = self.df[col].isna()
                num_failed = failed_mask.sum()
                print(f"[INFO] '{col}': {num_failed} rows failed default parsing out of {len(self.df)}")

                if num_failed > 0:
                    for fmt in alternative_formats:
                        try:
                            # Parse only failed rows with specific format
                            parsed_dates = pd.to_datetime(self.df.loc[failed_mask, col], format=fmt, errors='coerce')
                            success_mask = parsed_dates.notna()
                            self.df.loc[failed_mask, col] = parsed_dates
                            failed_mask = self.df[col].isna()
                            if failed_mask.sum() == 0:
                                print(f"[INFO] All dates in '{col}' parsed successfully using format {fmt}")
                                break
                        except Exception as e:
                            print(f"[WARNING] Error parsing '{col}' with format {fmt}: {e}")

                if self.df[col].isna().sum() > 0:
                    print(f"[WARNING] Some values in '{col}' could not be parsed into datetime:")
                    print(self.df.loc[self.df[col].isna(), col].head())
                if col == 'VehicleIntroDate':
                    self.df['VehicleIntroDate_missing'] = self.df[col].isna()

    def handle_missing_values(self):
        threshold = 0.9 * len(self.df)
        self.df.dropna(thresh=threshold, axis=1, inplace=True)

        if 'CustomValueEstimate' in self.df.columns:
            self.df['CustomValueEstimate'].fillna(self.df['CustomValueEstimate'].median(), inplace=True)

        for col in ['Gender', 'Bank', 'AccountType', 'MaritalStatus']:
            if col in self.df.columns:
                self.df[col].fillna('Unknown', inplace=True)

    def add_loss_ratio(self):
        if 'TotalClaims' in self.df.columns and 'TotalPremium' in self.df.columns:
            self.df['LossRatio'] = self.df['TotalClaims'] / self.df['TotalPremium']
            self.df.replace([np.inf, -np.inf], np.nan, inplace=True)

    def summary_statistics(self):
        print("Numeric data summary:")
        print(self.df.describe())

        print("\nCategorical data summary:")
        print(self.df.describe(include=['object', 'category']))

        print("\nDatetime columns summary:")
        date_cols = [col for col in ['TransactionMonth', 'VehicleIntroDate'] if col in self.df.columns]
        for col in date_cols:
            print(f"\nDatetime column summary: {col}")
            if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                try:
                    print(self.df[col].describe(datetime_is_numeric=True))
                except TypeError:
                    print("  datetime_is_numeric=True not supported — fallback to count/non-null:")
                    print(self.df[col].describe())
            else:
                print("  Warning: Column is not recognized as datetime or contains only NaT values.")
                print(f"  Sample values: {self.df[col].dropna().unique()[:5]}")

    def plot_univariate(self):
        if 'TotalPremium' in self.df.columns:
            self.df['TotalPremium'].hist(bins=50)
            plt.title('Distribution of TotalPremium')
            plt.xlabel('TotalPremium')
            plt.ylabel('Frequency')
            plt.show()

        if 'CoverType' in self.df.columns:
            self.df['CoverType'].value_counts().plot(kind='bar')
            plt.title('Distribution of CoverType')
            plt.xlabel('CoverType', fontsize=12)
            plt.ylabel('Count', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.show()

    def correlation_heatmap(self):
        corr = self.df.corr(numeric_only=True)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        plt.figure(figsize=(17, 15))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .75})
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.show()

    def temporal_trend(self):
        if 'TransactionMonth' in self.df.columns:
            self.df.set_index('TransactionMonth', inplace=True)
            trend = self.df.resample('ME')[['TotalPremium', 'TotalClaims']].sum()
            trend.plot(title='Monthly TotalPremium vs TotalClaims')
            plt.xlabel('Month')
            plt.ylabel('Amount')
            plt.tight_layout()
            plt.show()
            self.df.reset_index(inplace=True)  # restore index

    def loss_ratio_by_group(self, group_col):
        if 'LossRatio' in self.df.columns and group_col in self.df.columns:
            grouped = self.df.groupby(group_col)['LossRatio'].mean().sort_values()
            grouped.plot(kind='bar', title=f'Average Loss Ratio by {group_col}')
            plt.ylabel('Loss Ratio')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

    def boxplot_outliers(self, col):
        if col in self.df.columns:
            plt.figure(figsize=(8, 4))
            sns.boxplot(x=self.df[col])
            plt.title(f'Boxplot of {col}')
            plt.tight_layout()
            plt.show()

    def get_clean_data(self):
        return self.df
