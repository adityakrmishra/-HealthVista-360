from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import shap
import matplotlib.pyplot as plt
import joblib

def calculate_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }

def shap_analysis(model, X_train, X_test):
    explainer = shap.Explainer(model.named_steps['logisticregression'], 
                              model.named_steps['standardscaler'].transform(X_train))
    shap_values = explainer(model.named_steps['standardscaler'].transform(X_test))
    
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig('results/shap_plots/feature_importance.png', bbox_inches='tight')
    plt.close()
