import pandas as pd

class DataLoader:
    def __init__(self, filepath, delimiter='\t'):
        self.filepath = filepath
        self.delimiter = delimiter
        self.df = None

    def load(self):
        try:
            self.df = pd.read_csv(self.filepath, delimiter=self.delimiter, low_memory=False)
            return self.df
        except Exception as e:
            print(f"Error loading file: {e}")
            return None
