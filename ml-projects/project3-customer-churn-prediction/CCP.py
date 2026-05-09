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


df = pd.read_csv("churn.csv")# Load the customer churn dataset from a CSV file named "churn.csv" using pandas, which contains various features related to customer behavior 
#and a target variable indicating whether the customer has churned or not.   
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")# Convert the "TotalCharges" column to numeric values, coercing any non-numeric entries to NaN,
#which is necessary for proper analysis and modeling of this feature.
df.dropna(inplace=True)# Remove any rows with missing values from the DataFrame to ensure that the dataset is clean and complete for training machine learning models,
#as missing values can lead to errors or biased results during model training and evaluation.

# Encode categorical features using LabelEncoder, which converts categorical string values into numerical values that can be used by machine learning algorithms.
for col in df.select_dtypes(include=["object", "str"]).columns:
    df[col] = LabelEncoder().fit_transform(df[col])# Iterate through all columns in the DataFrame that have a data type of object or string, and apply LabelEncoder to each of these columns to transform the categorical values into numerical format,
    #allowing the machine learning models to process these features effectively during training and prediction.

X = df.drop("Churn", axis=1)# Define the feature matrix X by dropping the "Churn" column from the DataFrame, which contains the target variable indicating whether a customer has churned or not.
y = df["Churn"]# Define the target variable y as the "Churn" column from the DataFrame, which will be used to train the machine learning models to predict customer churn based on the features in X.

# Split the dataset into training and testing sets using the train_test_split function from scikit-learn, with a test size of 20% of the data, 
# stratifying by the target variable y to maintain the same proportion of classes in both sets, and setting a random state for reproducibility.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train and evaluate multiple machine learning models (Logistic Regression, Random Forest, and XGBoost) on the training data,
for name, model in [
    # Logistic Regression is implemented with a pipeline that includes standard scaling of features and logistic regression with a maximum of 2000 iterations and the "lbfgs" solver for optimization.
    ("Logistic Regression", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, solver="lbfgs"))), 
    # Random Forest is implemented with a random state for reproducibility, 
    # which is an ensemble learning method that builds multiple decision trees and merges their results to improve accuracy and control overfitting.
    ("Random Forest", RandomForestClassifier(random_state=42)),
    # XGBoost is implemented with a specified evaluation metric of "logloss" for binary classification and a random state for reproducibility,
    # which is an optimized gradient boosting library designed to be highly efficient, flexible, and portable, often used for structured data and known for its performance in machine learning competitions.
    ("XGBoost", XGBClassifier(eval_metric="logloss", random_state=42)),
]:
    model.fit(X_train, y_train)# Fit each model to the training data, allowing it to learn the relationships between the features in X_train 
    #and the target variable y_train (Churn) so that it can make predictions on new, unseen data.
    preds = model.predict(X_test)# Use the trained model to make predictions on the test set (X_test), which will be compared to the actual labels (y_test) 
    #to evaluate the model's performance in predicting customer churn.
    probs = model.predict_proba(X_test)[:, 1]# Get the predicted probabilities for the positive class (churn) from the model's predict_proba method, which returns an array of probabilities for each class,
    #and we select the probabilities for the positive class (churn) by taking the second column ([:, 1]) of the output, which will be used to calculate the AUC score for evaluating the model's performance in distinguishing between churned and non-churned customers.
    print(f"{name} -> F1: {f1_score(y_test, preds):.2f}, AUC: {roc_auc_score(y_test, probs):.2f}")# Print the F1 score and AUC score for each model, which are metrics used to evaluate the performance of the models in predicting customer churn.
    
#graphs      
plt.figure(figsize=(10, 6))# Set the size of the figure to 10 inches in width and 6 inches in height to provide enough space for visualizing the distribution of the target variable (Churn) in the dataset.
sns.countplot(x="Churn", data=df)# Create a count plot using seaborn to visualize the distribution of the "Churn" variable in the dataset, showing the count of customers who have churned (1) versus those who have not churned (0), which helps in understanding the class imbalance in the target variable.
plt.title("Distribution of Churn")# Set the title of the plot to "Distribution of Churn" to indicate that the plot is showing the distribution of the target variable (Churn) in the dataset, providing context for the visualization.
plt.show()# Display the plot to the user, allowing them to visually analyze the distribution of the Churn variable and understand the proportion of customers who have churned versus those who have not.
