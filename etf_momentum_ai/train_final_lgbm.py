import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMClassifier
from joblib import dump, load

# 1) Load best Optuna params:
best = load("params/optuna_lgbm_best_params.pkl")  # save study.best_params after hyperopt

df = pd.read_csv("datasets/ml_risk_v2_enhanced_regime.csv", index_col=0)
X, y = df.drop(columns=["Crash"]), df["Crash"]  

# 2) TimeSeriesSplit train all but last fold:
tscv = TimeSeriesSplit(n_splits=5)
splits = list(tscv.split(X))
train_idx, _ = zip(*splits[:-1])
train_idx = [i for fold in train_idx for i in fold]  # flatten first 4 folds
X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]

# 3) Fit base model:
base = LGBMClassifier(**best, verbosity=-1)
base.fit(X_tr, y_tr)

# 4) Calibrate:
cal = CalibratedClassifierCV(base, method="isotonic", cv=tscv)
cal.fit(X_tr, y_tr)

# 5) Save final:
dump(cal, "models/lgbm_crash_predictor.joblib")
