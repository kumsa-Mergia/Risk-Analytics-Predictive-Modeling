import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from scipy.stats import ttest_ind, f_oneway

class InsuranceEDA:
    def __init__(self, filepath, delimiter='|'):
        self.filepath = filepath
        self.delimiter = delimiter
        self.df = pd.read_csv(self.filepath, delimiter=self.delimiter, low_memory=False)
        self._convert_dates()
        self.test_results = []  # Store all hypothesis test summaries

    def load_data(self):
        self.df = pd.read_csv(self.filepath, delimiter=self.delimiter, low_memory=False)
        self._convert_dates()

    def _convert_dates(self):
        if 'TransactionMonth' in self.df.columns:
            self.df['TransactionMonth'] = pd.to_datetime(
                self.df['TransactionMonth'], format='%Y-%m-%d %H:%M:%S', errors='coerce'
            )

        if 'VehicleIntroDate' in self.df.columns:
            # Prepend "01/" to form "01/06/2002", then parse
            self.df['VehicleIntroDate'] = pd.to_datetime(
                '01/' + self.df['VehicleIntroDate'].astype(str), 
                format='%d/%m/%Y', 
                errors='coerce'
            )

    def add_date_parts(self):
        if 'TransactionMonth' in self.df.columns:
            self.df['TransactionYear'] = self.df['TransactionMonth'].dt.year
            self.df['TransactionQuarter'] = self.df['TransactionMonth'].dt.to_period('Q')
        if 'VehicleIntroDate' in self.df.columns:
            self.df['VehicleYear'] = self.df['VehicleIntroDate'].dt.year
            self.df['VehicleMonth'] = self.df['VehicleIntroDate'].dt.month


    def validate_dates(self):
        for col in ['TransactionMonth', 'VehicleIntroDate']:
            if col in self.df.columns:
                null_rate = self.df[col].isna().mean()
                print(f"{col}: {100 * null_rate:.2f}% missing after parsing.")


    def handle_missing_values(self):
        threshold = 0.9 * len(self.df)
        self.df.dropna(thresh=threshold, axis=1, inplace=True)
        
        if 'CustomValueEstimate' in self.df.columns:
            self.df['CustomValueEstimate'] = self.df['CustomValueEstimate'].fillna(self.df['CustomValueEstimate'].median())
        
        for col in ['Gender', 'Bank', 'AccountType', 'MaritalStatus']:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna('Unknown')

    def add_metrics(self):
        if 'TotalClaims' in self.df.columns and 'TotalPremium' in self.df.columns:
            self.df['LossRatio'] = self.df['TotalClaims'] / self.df['TotalPremium']
            self.df['ClaimFrequency'] = self.df['TotalClaims'] > 0
            self.df['ClaimSeverity'] = self.df.apply(
                lambda row: row['TotalClaims'] / row['ClaimCount'] if row.get('ClaimCount', 0) > 0 else np.nan, axis=1)
            self.df['Margin'] = self.df['TotalPremium'] - self.df['TotalClaims']
            self.df.replace([np.inf, -np.inf], np.nan, inplace=True)

    def check_equivalence(self, group_col, feature):
        # Quick check if groups are balanced on feature
        ctab = pd.crosstab(self.df[group_col], self.df[feature], normalize='index')
        print(f"\nGroup Equivalence Check: {group_col} vs {feature}\n{ctab.head()}")
        return ctab

    def perform_ttest(self, metric, group_col, group1, group2, hypothesis_name):
        g1 = self.df[self.df[group_col] == group1][metric].dropna()
        g2 = self.df[self.df[group_col] == group2][metric].dropna()

        if len(g1) < 30 or len(g2) < 30:
            print(f"Not enough data for t-test on {group_col} ({group1} vs {group2})")
            return

        stat, p = ttest_ind(g1, g2, equal_var=False)
        decision = "Reject H₀" if p < 0.05 else "Fail to Reject H₀"
        print(f"\nT-test on {metric} by {group_col} ({group1} vs {group2})")
        print(f"t = {stat:.4f}, p = {p:.4f} → {decision}")

        interpretation = f"There is a {'significant' if p < 0.05 else 'no significant'} difference in {metric.lower()} between {group1} and {group2}."

        self.test_results.append({
            'Hypothesis': hypothesis_name,
            'Metric': metric,
            'Group': f"{group1} vs {group2}",
            'p-value': round(p, 4),
            'Decision': decision,
            'Interpretation': interpretation
        })

    def perform_anova(self, metric, group_col, hypothesis_desc="", min_group_size=30):
        valid_groups = []
        group_labels = []
        for name, grp in self.df.groupby(group_col):
            data = grp[metric].dropna()
            if len(data) >= min_group_size:
                valid_groups.append(data)
                group_labels.append(name)

        if len(valid_groups) < 2:
            print(f" Not enough valid groups for ANOVA on {group_col}")
            return

        stat, p = f_oneway(*valid_groups)
        
        decision = "Reject H₀" if p < 0.05 else "Fail to Reject H₀"
        interpretation = (
            f"There is a {'significant' if p < 0.05 else 'no significant'} difference in {metric.lower()} across {group_col} groups."
        )
        print(f"\nANOVA: {metric} by {group_col}")
        print(f"F-statistic = {stat:.4f}, p-value = {p:.4f}")
        print(f" {decision} – {hypothesis_desc}")
        
        #  Append to results summary
        self.test_results.append({
            'Hypothesis': hypothesis_desc,
            'Metric': metric,
            'Group': f"{group_col} ({len(group_labels)} groups)",
            'p-value': round(p, 4),
            'Decision': decision,
            'Interpretation': interpretation
    })
        
        # if p < 0.05:
        #     print(f" Reject H₀ – {hypothesis_desc}")
        # else:
        #     print(f" Fail to Reject H₀ – {hypothesis_desc}")

    def hypothesis_tests(self):
        print("\nStarting Hypothesis Testing...\n")
        self.add_metrics()

        # H₀: No risk differences across provinces
        if 'Province' in self.df.columns:
            self.perform_anova(
                metric='LossRatio', 
                group_col='Province', 
                hypothesis_desc="No risk differences across provinces"
            )


        # H₀: No risk differences between Postal codes
        if 'PostalCode' in self.df.columns:
            self.perform_anova(
                metric='LossRatio', 
                group_col='PostalCode', 
                hypothesis_desc="No risk differences between postal codes"
                )

        # H₀: No significant margin difference between Postal codes
        if 'PostalCode' in self.df.columns:
            self.perform_anova(
                metric='Margin', 
                group_col='PostalCode', 
                hypothesis_desc="No significant margin difference between postal codes"
                )

        # H₀: No significant risk difference between Women and Men
        if 'Gender' in self.df.columns:
            self.perform_ttest(
                metric='LossRatio', 
                group_col='Gender', 
                group1='Male', 
                group2='Female',
                hypothesis_name="No significant risk difference between Women and Men"
                )

    def report_results(self):
        print("\nSummary of Hypothesis Test Results:")
        return pd.DataFrame(self.test_results)

    def visualize_boxplot(self, metric, group_col, top_n=20):
        df_plot = self.df[[group_col, metric]].dropna()

    # Handle high cardinality: keep only top_n categories
        group_order = (
            df_plot.groupby(group_col)[metric]
            .median()
            .sort_values(ascending=False)
            .head(top_n)
            .index.tolist()
        )

        plt.figure(figsize=(12, 6))
        sns.boxplot(
            data=df_plot[df_plot[group_col].isin(group_order)],
            x=group_col,
            y=metric,
            hue=group_col,           # Fix for future Seaborn
            order=group_order,
            palette="Set2",
            legend=False             # Avoid duplicate legend for hue=x
        )

        plt.title(f"{metric} Distribution by {group_col}", fontsize=14)
        plt.xlabel(group_col, fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.xticks(rotation=30, ha='right')
        plt.grid(True, axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()


    def get_clean_data(self):
        return self.df