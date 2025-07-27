import optuna, pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import fbeta_score, make_scorer
from lightgbm import LGBMClassifier
import lightgbm as lgb
import joblib

# 1. Load data
df = pd.read_csv("datasets/ml_risk_v2_enhanced_regime.csv", index_col=0)
X = df.drop(columns=["Crash"]) 
y = df["Crash"]

tscv = TimeSeriesSplit(n_splits=5)
f2 = make_scorer(fbeta_score, beta=2)

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.005, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 16, 256),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200),
        "feature_fraction": trial.suggest_uniform("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.4, 1.0),
        "class_weight": {0:1, 1:trial.suggest_int("pos_weight", 3, 20)},
        "random_state": 42,
        "n_jobs": -1
    }
    cv_scores = []
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = LGBMClassifier(**params, verbosity=-1)
        # use the callbacks API for early stopping
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50)],
        )
        preds = model.predict_proba(X_val)[:,1]
        # pick threshold that maximizes F2:
        best = 0
        for thr in [i/100 for i in range(1,100)]:
            score = fbeta_score(y_val, (preds>=thr).astype(int), beta=2)
            best = max(best, score)
        cv_scores.append(best)
    return sum(cv_scores)/len(cv_scores)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, timeout=1800)
print("Best F2:", study.best_value)
print(study.best_params)
joblib.dump(study.best_params, "params/optuna_lgbm_best_params.pkl")
print("Saved best params to params/optuna_lgbm_best_params.pkl")