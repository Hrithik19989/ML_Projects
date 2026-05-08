import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import seaborn as sns 

df = pd.read_csv("churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

for col in df.select_dtypes(include=["object", "str"]).columns:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

for name, model in [
    ("Logistic Regression", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, solver="lbfgs"))),
    ("Random Forest", RandomForestClassifier(random_state=42)),
    ("XGBoost", XGBClassifier(eval_metric="logloss", random_state=42)),
]:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    print(f"{name} -> F1: {f1_score(y_test, preds):.2f}, AUC: {roc_auc_score(y_test, probs):.2f}")
    
#graphs      
plt.figure(figsize=(10, 6))
sns.countplot(x="Churn", data=df)
plt.title("Distribution of Churn")
plt.show()
