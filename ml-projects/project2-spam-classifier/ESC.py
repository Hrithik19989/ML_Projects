import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import  ConfusionMatrixDisplay , classification_report , confusion_matrix
import matplotlib.pyplot as plt

df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "text"]
df["label"] = df["label"].map({"ham": 0, "spam": 1})

X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2)

tfidf = TfidfVectorizer(stop_words="english", max_features=3000)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

for name, model in [("Naive Bayes", MultinomialNB()),
                     ("Logistic Regression", LogisticRegression()),
                     ("SVM", SVC())]:
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)
    report = classification_report(
        y_test,
        preds,
        target_names=["Ham", "Spam"],
        output_dict=True,
    )

    print(
        f"\n{name}:\n",
        classification_report(y_test, preds, target_names=["Ham", "Spam"]),
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{name} Performance", fontsize=14)

    cm = confusion_matrix(y_test, preds)
    ConfusionMatrixDisplay(cm, display_labels=["Ham", "Spam"]).plot(
        ax=axes[0],
        cmap="Blues",
        colorbar=False,
    )
    axes[0].set_title("Confusion Matrix")

    metrics = ["precision", "recall", "f1-score"]
    x = range(len(metrics))
    ham_scores = [report["Ham"][metric] for metric in metrics]
    spam_scores = [report["Spam"][metric] for metric in metrics]

    axes[1].bar([i - 0.2 for i in x], ham_scores, width=0.4, label="Ham")
    axes[1].bar([i + 0.2 for i in x], spam_scores, width=0.4, label="Spam")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(["Precision", "Recall", "F1-score"])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Class Metrics")
    axes[1].set_ylabel("Score")
    axes[1].legend()

    plt.tight_layout()
    plt.show()