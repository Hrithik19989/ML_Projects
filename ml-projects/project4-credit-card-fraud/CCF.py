import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ==========================
# Load Dataset
# ==========================
df = pd.read_csv("cleaned_creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

# ==========================
# Isolation Forest
# ==========================
print("="*60)
print("Isolation Forest")

iso = IsolationForest(
    contamination=0.0017,
    random_state=42
)

iso.fit(X)

# decision_function:
# higher score = normal
# lower score = anomaly
iso_scores = iso.decision_function(X)

# ==========================
# Threshold Tuning
# ==========================

thresholds = np.percentile(iso_scores, [0.1,0.2,0.3,0.5,1,2,3,5])

best_f1 = 0
best_threshold = None
best_prediction = None

print("\nThreshold Tuning")

for t in thresholds:

    pred = (iso_scores < t).astype(int)

    precision = precision_score(y, pred)
    recall = recall_score(y, pred)
    f1 = f1_score(y, pred)

    print(f"Threshold={t:.5f}  Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t
        best_prediction = pred

print("\nBest Threshold:", best_threshold)

print(classification_report(y, best_prediction))

# ==========================
# One-Class SVM
# ==========================

print("="*60)
print("One-Class SVM")

svm = OneClassSVM(
    kernel="rbf",
    gamma="scale",
    nu=0.0017
)

svm.fit(X)

svm_scores = svm.decision_function(X)

thresholds = np.percentile(svm_scores,[0.1,0.2,0.3,0.5,1,2,3,5])

best_f1_svm = 0
best_threshold_svm = None
best_prediction_svm = None

print("\nThreshold Tuning")

for t in thresholds:

    pred = (svm_scores < t).astype(int)

    precision = precision_score(y,pred)
    recall = recall_score(y,pred)
    f1 = f1_score(y,pred)

    print(f"Threshold={t:.5f} Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f}")

    if f1 > best_f1_svm:
        best_f1_svm = f1
        best_threshold_svm = t
        best_prediction_svm = pred

print("\nBest Threshold:", best_threshold_svm)

print(classification_report(y,best_prediction_svm))

# ==========================
# Comparison
# ==========================

comparison = pd.DataFrame({
    "Model":["Isolation Forest","One-Class SVM"],
    "Precision":[
        precision_score(y,best_prediction),
        precision_score(y,best_prediction_svm)
    ],
    "Recall":[
        recall_score(y,best_prediction),
        recall_score(y,best_prediction_svm)
    ],
    "F1 Score":[
        f1_score(y,best_prediction),
        f1_score(y,best_prediction_svm)
    ]
})

print("\nModel Comparison")
print(comparison)

# ==========================
# Confusion Matrix
# ==========================

fig,ax = plt.subplots(1,2,figsize=(12,5))

sns.heatmap(
    confusion_matrix(y,best_prediction),
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=ax[0]
)
ax[0].set_title("Isolation Forest")

sns.heatmap(
    confusion_matrix(y,best_prediction_svm),
    annot=True,
    fmt="d",
    cmap="Greens",
    ax=ax[1]
)
ax[1].set_title("One-Class SVM")

plt.tight_layout()
plt.show()

# ==========================
# Precision vs Recall
# ==========================

comparison.set_index("Model")[["Precision","Recall"]].plot(
    kind="bar",
    figsize=(7,5)
)

plt.title("Precision vs Recall")
plt.ylabel("Score")
plt.ylim(0,1)
plt.grid(axis="y")

plt.show()