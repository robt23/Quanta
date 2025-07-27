import pandas as pd
import numpy as np

from sklearn.model_selection    import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing      import StandardScaler
from sklearn.metrics            import make_scorer, fbeta_score, precision_recall_curve, classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling     import SMOTE
from imblearn.pipeline          import Pipeline
from lightgbm                   import LGBMClassifier
from sklearn.calibration        import CalibratedClassifierCV, calibration_curve
import matplotlib.pyplot as plt

# 1) Load data
df = pd.read_csv("datasets/ml_risk_dataset_regime.csv", index_col=0, parse_dates=True)
X, y = df.drop(columns=["Label"]), df["Label"]

split_idx = int(len(df) * 0.7)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# 2) Pipeline: scale → SMOTE → LGBM with heavy positive class weight
pipeline = Pipeline([
    ("scale", StandardScaler()),                   # zero-mean/unit-variance scaling
    ("smote", SMOTE(random_state=42)),             # synthetically balance your minority class
    ("lgbm", LGBMClassifier(
        class_weight={0:1, 1:10},                  # overweight crashes by 10×
        random_state=42
    ))
])

# 3) Define F2 scorer
f2_scorer = make_scorer(fbeta_score, beta=2)

# 4) TimeSeriesSplit for CV
tscv = TimeSeriesSplit(n_splits=3)
param_grid = {
    "lgbm__n_estimators":    [100, 200, 300],
    "lgbm__max_depth":       [3, 5, 7, 15, None],
    "lgbm__learning_rate":   [0.01, 0.1],
    "lgbm__min_child_samples": [1, 5, 10, 20]
}

# 5) GridSearchCV optimizing for F2 score
grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring=f2_scorer,    # <-- directly optimize F₂
    cv=tscv,
    n_jobs=-1,
    verbose=2
)

grid.fit(X_train, y_train)
print("Best CV F₂:", grid.best_score_)
print("Best params:", grid.best_params_)

# 6) Evaluate on test set
best_pipe = grid.best_estimator_
y_proba_raw = best_pipe.predict_proba(X_test)[:,1]
print("Hold-out ROC AUC:", roc_auc_score(y_test, y_proba_raw))
print(classification_report(y_test, best_pipe.predict(X_test)))
print("Confusion matrix:\n", confusion_matrix(y_test, best_pipe.predict(X_test)))

# 7) Calibrate with Platt scaling
calibrator = CalibratedClassifierCV(best_pipe, method="sigmoid", cv=tscv)
calibrator.fit(X_train, y_train)
y_proba_calib = calibrator.predict_proba(X_test)[:,1]

# 8) Plot precision-recall curve and find best F₂ threshold
probs = y_proba_calib  # or y_proba_raw
precision, recall, thresholds = precision_recall_curve(y_test, probs)

# Compute F₂ for each threshold
f2_scores = (1+2**2) * (precision * recall) / (2**2 * precision + recall + 1e-12)
best_idx   = np.argmax(f2_scores[:-1])  
best_thresh= thresholds[best_idx]

print(f"Best F₂={f2_scores[best_idx]:.3f} at threshold={best_thresh:.3f}")

# 9) Final predictions using the best threshold
y_pred = (probs >= best_thresh).astype(int)
print(classification_report(y_test, y_pred, zero_division=0))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# 10) Inspect feature importances
importances = pd.Series(
    best_pipe.named_steps["lgbm"].feature_importances_,
    index=X.columns
).sort_values(ascending=False)
print("Feature importances:\n", importances)
