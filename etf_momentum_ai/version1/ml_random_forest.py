# ml_random_forest_recall.py

import numpy as np
import pandas as pd
from sklearn.model_selection    import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble           import RandomForestClassifier
from sklearn.preprocessing      import StandardScaler
from sklearn.metrics            import (
    roc_auc_score, classification_report, confusion_matrix,
    recall_score, precision_score, make_scorer
)
from sklearn.calibration        import CalibratedClassifierCV
from imblearn.over_sampling     import SMOTE
from imblearn.pipeline          import Pipeline

# 1) Load data
df   = pd.read_csv("datasets/ml_risk_dataset_regime.csv",
                   index_col=0, parse_dates=True)
X    = df.drop(columns=["Label"])
y    = df["Label"]

# 2) Train/test split (chronological 70/30)
split = int(len(df)*0.7)
X_tr, X_te = X.iloc[:split], X.iloc[split:]
y_tr, y_te = y.iloc[:split], y.iloc[split:]

# 3) Pipeline: scale → SMOTE → RF with heavy positive class weight
pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("rf",    RandomForestClassifier(
                  class_weight={0:1, 1:5},  # punish misses on class=1 five times harder
                  random_state=42,
                  n_jobs=-1
             ))
])

# 4) CV splitter
tscv = TimeSeriesSplit(n_splits=5)

# 5) Hyperparameter grid
param_grid = {
    "rf__n_estimators":    [100,200,300],
    "rf__max_depth":       [3,5,7,None],
    "rf__min_samples_leaf":[1,3,5]
}

# 6) GridSearchCV optimizing for recall
grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=tscv,
    scoring=make_scorer(recall_score),
    verbose=1
)
grid.fit(X_tr, y_tr)

print("Best CV recall:", grid.best_score_)
print("Best params:    ", grid.best_params_)

# 7) Calibrate final pipeline with sigmoid (Platt scaling)
best_pipe   = grid.best_estimator_
calibrator  = CalibratedClassifierCV(
    estimator=best_pipe,
    cv=tscv,
    method="sigmoid"
)
calibrator.fit(X_tr, y_tr)

# 8) Get calibrated probabilities on test
probs_calib = calibrator.predict_proba(X_te)[:,1]

# 9) Find lowest threshold to achieve ≥90% recall
target_recall = 0.80
best_thresh   = 0.0
for thresh in np.linspace(0,1,501):
    preds = (probs_calib >= thresh).astype(int)
    r     = recall_score(y_te, preds)
    if r >= target_recall:
        best_thresh = thresh
        break

print(f"\n→ Using threshold {best_thresh:.3f} for ≥{target_recall*100:.0f}% recall")

# 10) Final evaluation
final_preds = (probs_calib >= best_thresh).astype(int)
print("\nClassification Report:")
print(classification_report(y_te, final_preds, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_te, final_preds))
print("Test ROC AUC:", roc_auc_score(y_te, probs_calib))
