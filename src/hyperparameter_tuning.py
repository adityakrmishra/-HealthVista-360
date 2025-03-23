"""
Advanced Hyperparameter Optimization
- Uses Optuna for Bayesian optimization
- Supports distributed tuning
- Integrates with model registry
"""
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score
from .models import save_model
from .utils import load_config

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 0.5),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
    }
    
    model = XGBClassifier(**params)
    score = cross_val_score(
        model, X_train, y_train,
        cv=5, scoring='roc_auc'
    ).mean()
    
    return score

def optimize_hyperparameters(n_trials=100):
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42)
    
    study.optimize(objective, n_trials=n_trials)
    
    # Save best model
    best_model = XGBClassifier(**study.best_params)
    best_model.fit(X_train, y_train)
    save_model(best_model, "models/optimized_xgb.pkl")
    
    return study.best_params
