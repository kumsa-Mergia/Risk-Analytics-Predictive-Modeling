import pandas as pd

class DataLoader:
    def __init__(self, filepath, delimiter='|'):
        self.filepath = filepath
        self.delimiter = delimiter

    def load(self):
        return pd.read_csv(self.filepath, delimiter=self.delimiter, low_memory=False)
