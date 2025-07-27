# ml_train.py

# 1. Imports
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve
)
from imblearn.over_sampling import SMOTE

# 2. Load the risk dataset
df = pd.read_csv("datasets/ml_risk_dataset_regime.csv", index_col=0, parse_dates=True)

# 3. Separate features (X) and label (y)
X = df.drop(columns=["Label"])
y = df["Label"]                 # 1 if >5% SPY drop ahead, else 0

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = 0.3,
    random_state = 42,
    stratify     = y
)

# 5. Scale features
scaler        = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# 5a. Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
print("After SMOTE, counts of label '1':", sum(y_train_balanced==1), 
      " vs label '0':", sum(y_train_balanced==0))

# 6. Train Logistic Regression
clf = LogisticRegression(class_weight="balanced", random_state=42)
clf.fit(X_train_balanced, y_train_balanced)

# 7. Predict probabilities
y_proba = clf.predict_proba(X_test_scaled)[:, 1]



# === SECTION A: Fixed‐threshold evaluation ===

chosen_thresh = 0.34  # your fixed cutoff

# Hard predictions at fixed threshold
y_pred_fixed = (y_proba >= chosen_thresh).astype(int)

print(f"\n--- Metrics at fixed threshold = {chosen_thresh:.3f} ---")
print("Classification Report:\n", classification_report(y_test, y_pred_fixed))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_fixed))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba))



# === SECTION B: Dynamic‐threshold tuning for a desired recall ===

# 1. Compute precision, recall for all possible thresholds
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

# 2. Plot Precision–Recall vs Threshold
plt.figure(figsize=(8, 5))
plt.plot(thresholds, precisions[:-1], label="Precision", linewidth=2)
plt.plot(thresholds, recalls[:-1],    label="Recall",    linewidth=2)
plt.xlabel("Threshold for predicting crash")
plt.ylabel("Precision / Recall")
plt.title("Precision–Recall vs. Threshold")
plt.legend()
plt.grid(True)
plt.show()

# 3. Specify the recall you want to achieve
target_recall = 0.80  # e.g. 80% of crashes

# 4. Find all thresholds giving at least target_recall
valid_idxs = [i for i, r in enumerate(recalls[:-1]) if r >= target_recall]
if not valid_idxs:
    raise ValueError(f"No threshold achieves recall ≥ {target_recall:.2f}")

# 5. Pick the highest threshold among those (to maximize precision under your recall constraint)
best_idx    = max(valid_idxs)
best_thresh = thresholds[best_idx]
best_prec   = precisions[best_idx]
best_rec    = recalls[best_idx]

print(f"\n--- To get recall ≥ {target_recall:.0%}, use threshold ≈ {best_thresh:.3f} ---")
print(f" At that cutoff: precision = {best_prec:.3f}, recall = {best_rec:.3f}")

# 6. Evaluate at this “best_thresh”
y_pred_dyn = (y_proba >= best_thresh).astype(int)

print("\nClassification Report at dynamic threshold:\n",
      classification_report(y_test, y_pred_dyn))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dyn))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba))
