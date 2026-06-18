import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt 
import seaborn as sns

# Load the credit card fraud dataset from a CSV file named "creditcard.csv" using pandas.
# The dataset contains various features related to credit card transactions, and the target variable "Class" indicates
# whether a transaction is fraudulent (1) or not (0).
df = pd.read_csv("creditcard.csv")
X = df.drop("Class", axis=1)
y = df["Class"]

iso = IsolationForest(contamination=0.01, random_state=42)
preds = iso.fit_predict(X)
preds = [0 if p == 1 else 1 for p in preds]  # convert to 0/1
print("Isolation Forest:\n", classification_report(y, preds))

# Visualize original class distribution
plt.figure(figsize=(6, 4))

sns.countplot(x=y)

plt.title("Original Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")

plt.show()

# Visualize predicted class distribution
plt.figure(figsize=(6, 4))

sns.countplot(x=preds)

plt.title("Predicted Fraud Distribution")
plt.xlabel("Predicted Class")
plt.ylabel("Count")

plt.show()
