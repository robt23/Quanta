import joblib
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    auc,
    fbeta_score,
    precision_score,
    recall_score
)
from sklearn.metrics import roc_auc_score, average_precision_score
import json


# 1) Load the trained model
model = joblib.load("models/lgbm_crash_predictor.joblib")

# 2) Load your final dataset and split off the last 20% as hold-out
df    = pd.read_csv("datasets/ml_risk_v2_enhanced_regime.csv", index_col=0)
split = int(len(df) * 0.8)
X_te  = df.drop(columns=["Crash"]).iloc[split:]
y_te  = df["Crash"].iloc[split:]

# 3) Get predicted probabilities and hard predictions at your chosen threshold
probs = model.predict_proba(X_te)[:, 1]
th = json.load(open("models/crash_threshold.json"))["threshold"]
preds = (probs >= th).astype(int)

# 4) Confusion matrix
cm = confusion_matrix(y_te, preds)
print("Confusion matrix (TN, FP; FN, TP):")
print(cm, "\n")

# 5) Precision–Recall AUC & ROC AUC
precision, recall, _ = precision_recall_curve(y_te, probs)
pr_auc = auc(recall, precision)
roc_auc = roc_auc_score(y_te, probs)
print(f"PR AUC   = {pr_auc:.3f}")
print(f"ROC AUC  = {roc_auc:.3f}\n")

# ▶ Added: alternative PR AUC via average_precision_score
avg_pr = average_precision_score(y_te, probs)
print(f"Avg Precision–Recall AUC = {avg_pr:.3f}\n")

# 6) F2 score at threshold
f2 = fbeta_score(y_te, preds, beta=2)
print(f"F₂ score @ threshold {th:.3f} = {f2:.3f}\n")

# 7) Print precision & recall again for clarity
print(f"At threshold {th:.3f}:")
print(f"  Precision = {precision_score(y_te, preds):.3f}")
print(f"  Recall    = {recall_score(y_te, preds):.3f}")
