from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

class ModelBuilder:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def train_linear_regression(self):
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        return model

    def train_random_forest(self):
        model = RandomForestRegressor(random_state=42)
        model.fit(self.X_train, self.y_train)
        return model

    def train_xgboost(self):
        model = xgb.XGBRegressor(random_state=42, use_label_encoder=False, eval_metric='rmse')
        model.fit(self.X_train, self.y_train)
        return model
