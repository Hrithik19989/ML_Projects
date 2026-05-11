import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt 

df = pd.read_csv("creditcard.csv")
X = df.drop("Class", axis=1)
y = df["Class"]

iso = IsolationForest(contamination=0.01, random_state=42)
preds = iso.fit_predict(X)
preds = [0 if p == 1 else 1 for p in preds]  # convert to 0/1
print("Isolation Forest:\n", classification_report(y, preds))

#Visualization the distribution of the target variable (Class) in the dataset, showing the count of fraudulent transactions (1) versus non-fraudulent transactions (0),
# which helps in understanding the class imbalance in the target variable.
