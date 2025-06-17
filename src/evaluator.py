from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

class Evaluator:
    def __init__(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test
        self.results = {}

    def evaluate_model(self, model, name):
        preds = model.predict(self.X_test)
        rmse = np.sqrt(mean_squared_error(self.y_test, preds))
        r2 = r2_score(self.y_test, preds)
        self.results[name] = {'RMSE': rmse, 'R2': r2}
        print(f"Evaluated {name}: RMSE={rmse:.4f}, R2={r2:.4f}")

    def compare_models(self):
        import pandas as pd
        df = pd.DataFrame(self.results).T
        print("\nModel comparison:")
        print(df)

