import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import  ConfusionMatrixDisplay , classification_report , confusion_matrix
import matplotlib.pyplot as plt

#Extracting and preprocessing the dataset
df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]# Load the dataset from a CSV file named "spam.csv" using pandas.
#The encoding is set to "latin-1" to handle any special characters in the text data.
df.columns = ["label", "text"]# Rename the columns of the DataFrame to "label" and "text" for better clarity and understanding of the dataset's structure.
df["label"] = df["label"].map({"ham": 0, "spam": 1})# Convert the "label" column from categorical values ("ham" and "spam") to numerical values (0 for ham and 1 for spam) using the map function, which is necessary for training machine learning models.

X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2)# Split the dataset into training and testing sets using the train_test_split function from scikit-learn.

tfidf = TfidfVectorizer(stop_words="english", max_features=3000)# Initialize a TfidfVectorizer to convert the text data into numerical features.
#The stop_words parameter is set to "english" to remove common English words that may not contribute to the classification, 
#and max_features is set to 3000 to limit the number of features to the top 3000 most important ones based on term frequency-inverse document frequency (TF-IDF) scores.
X_train_vec = tfidf.fit_transform(X_train)# Fit the TfidfVectorizer to the training text data and transform it into a sparse matrix of TF-IDF features,
#which will be used for training the machine learning models.
X_test_vec = tfidf.transform(X_test)# Transform the testing text data using the same TfidfVectorizer fitted on the training data,
#ensuring that the test features are represented in the same way as the training features.

for name, model in [("Naive Bayes", MultinomialNB()),
                     ("Logistic Regression", LogisticRegression()),
                     ("SVM", SVC())]:
    model.fit(X_train_vec, y_train)# Fit each model to the training data, allowing it to learn the relationships between the TF-IDF features and the target labels (ham or spam).
    preds = model.predict(X_test_vec)# Use the trained model to make predictions on the test set, which will be compared to the actual labels to evaluate the model's performance.
    report = classification_report(
        y_test,
        preds,
        target_names=["Ham", "Spam"],# Generate a classification report that includes precision, recall,
        #and F1-score for each class (Ham and Spam) based on the true labels (y_test) and the predicted labels (preds).
        output_dict=True,# The output_dict parameter is set to True to return the classification report as a dictionary,
        #which allows for easier access to the individual metrics for each class when visualizing the results.
    )

    print(
        f"\n{name}:\n",
        classification_report(y_test, preds, target_names=["Ham", "Spam"]),# Print the classification report for each model, showing the precision, recall, and F1-score 
        #for both Ham and Spam classes, which helps in evaluating the performance of each model in classifying the text messages correctly.
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))# Create a figure with two subplots side by side to visualize the performance of each model. 
    #The first subplot will display the confusion matrix, and the second subplot will show a bar chart of the precision, recall, and F1-score for both classes.
    fig.suptitle(f"{name} Performance", fontsize=14)# Set the overall title of the figure to indicate which model's performance is being visualized, 
    #providing context for the confusion matrix and class metrics displayed in the subplots.

    cm = confusion_matrix(y_test, preds)# Compute the confusion matrix for the true labels (y_test) and the predicted labels (preds), 
    #which will be used to visualize the number of true positives, true negatives, false positives, and false negatives for each class.
    ConfusionMatrixDisplay(cm, display_labels=["Ham", "Spam"]).plot(
        ax=axes[0],# Display the confusion matrix in the first subplot using ConfusionMatrixDisplay from scikit-learn, with labels "Ham" and "Spam" for the axes.
        cmap="Blues",# Use a blue color map to visually differentiate the values in the confusion matrix, where higher values will be shown in darker shades of blue.
        colorbar=False,# Disable the color bar for the confusion matrix to keep the visualization clean and focused on the values in the matrix itself.
    )
    axes[0].set_title("Confusion Matrix")# Set the title of the first subplot to "Confusion Matrix" to indicate that it displays the confusion matrix for the model's predictions.

    metrics = ["precision", "recall", "f1-score"]# Define a list of metrics (precision, recall, and F1-score) that will be visualized in the second subplot as bar charts for both Ham and Spam classes.
    x = range(len(metrics))# Create a range of x values corresponding to the number of metrics, which will be used as the positions for the bars in the bar chart for both classes.
    ham_scores = [report["Ham"][metric] for metric in metrics]# Extract the precision, recall, and F1-score for the Ham class from the classification
    #report dictionary and store them in a list (ham_scores) to be used for plotting the bar chart for the Ham class.
    spam_scores = [report["Spam"][metric] for metric in metrics]# Extract the precision, recall, and F1-score for the Spam class from the classification report dictionary and 
    #store them in a list (spam_scores) to be used for plotting the bar chart for the Spam class.

    axes[1].bar([i - 0.2 for i in x], ham_scores, width=0.4, label="Ham")# Plot a bar chart for the Ham class metrics (precision, recall, F1-score) in the second subplot, with bars positioned slightly to the left of the x-ticks for better visibility.
    axes[1].bar([i + 0.2 for i in x], spam_scores, width=0.4, label="Spam")# Plot a bar chart for the Spam class metrics (precision, recall, F1-score) in the second subplot,
    #with bars positioned slightly to the right of the x-ticks to differentiate them from the Ham class bars.
    axes[1].set_xticks(list(x))# Set the x-ticks of the second subplot to correspond to the positions of the metrics (precision, recall, F1-score) for both classes, allowing for clear labeling of the bars in the bar chart.
    axes[1].set_xticklabels(["Precision", "Recall", "F1-score"])# Set the x-tick labels of the second subplot to "Precision", "Recall", and "F1-score" to indicate which metric each bar represents for both Ham and Spam classes.
    axes[1].set_ylim(0, 1.05)# Set the y-axis limits of the second subplot to range from 0 to 1.05, which allows for a clear visualization of the precision, recall, and F1-score values for both classes, as these metrics typically range between 0 and 1.
    axes[1].set_title("Class Metrics")# Set the title of the second subplot to "Class Metrics" to indicate that it displays the precision, recall, and F1-score for both Ham and Spam classes, providing insight into the model's performance in classifying the text messages.
    axes[1].set_ylabel("Score")# Set the y-axis label of the second subplot to "Score" to indicate that the vertical axis represents the values of precision, recall, and F1-score for both classes.
    axes[1].legend()# Add a legend to the second subplot to differentiate between the bars representing the Ham class and the Spam class metrics, allowing for easier interpretation of the bar chart.

    plt.tight_layout()# Adjust the layout of the figure to ensure that the subplots and their titles, labels, and legends do not overlap, providing a clear and organized visualization of the model's performance metrics.ii
    plt.show()# Display the figure with the confusion matrix and class metrics for each model, allowing for a visual comparison of the performance of the Naive Bayes, Logistic Regression, and SVM models in classifying the text messages as Ham or Spam.  