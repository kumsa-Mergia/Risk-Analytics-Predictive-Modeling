import shap

class Explainer:
    def __init__(self, model, X_sample):
        self.model = model
        self.X_sample = X_sample
        self.explainer = shap.Explainer(model)

    def explain(self):
        shap_values = self.explainer(self.X_sample)
        shap.summary_plot(shap_values, self.X_sample)
