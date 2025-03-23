import shap
import matplotlib.pyplot as plt
import joblib

def generate_shap_summary(model_path, X_test, output_path):
    model = joblib.load(model_path)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
