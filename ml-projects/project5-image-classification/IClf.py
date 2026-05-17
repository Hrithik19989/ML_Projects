from sklearn.datasets import load_digits # load-digits dataset
from sklearn.model_selection import train_test_split #train test split module
from sklearn.linear_model import LogisticRegression #Logistic Regression
from sklearn.ensemble import RandomForestClassifier #Random Forest Classifier
from sklearn.metrics import accuracy_score, classification_report # accuracy score and classifcation report
import matplotlib.pyplot as plt #matplotlib
# Load the data & train the data 
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# Models for pattern from train / test data 
# Models are Logistic Regression and Random Forest 
models = [
    ("Logistic Regression", LogisticRegression(max_iter=1000, n_jobs=None)),
    ("Random Forest", RandomForestClassifier(random_state=42)),
]

# Empty list for storing  accuracy results for the models mentioned above 
accuracies = []

# Loop for models train test data fitting  , appending accuracy in the empty list above 
#print accuracy an classification report  
 
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

    print(f"{name} → Accuracy: {acc:.4f}")
    print(f"{name} Classification Report:\n{classification_report(y_test, y_pred)}\n")

# Graph for model accuracy (bar chart comparison)
model_names = [name for name, _ in models]

plt.figure(figsize=(10, 6))
colors = ["grey","black"]
plt.bar(model_names, accuracies, color=colors)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.ylim(0, 1.0)
#Accuracy on top of the bar
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom")
    
plt.tight_layout()
plt.show()

