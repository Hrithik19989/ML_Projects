import pandas as pd
from sklearn.model_selection import train_test_split # 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt



# Load the dataset from a CSV file named "Real-estate1.csv" using pandas.
df = pd.read_csv("Real-estate1_cleaned.csv")
df = df.select_dtypes(include=[np.number]).dropna()# Select only numeric columns and drop rows with missing values to ensure the dataset is clean for modeling.
print(df.head())# Display the first few rows of the dataset to understand its structure and contents
print(df.describe())# Get summary statistics of the dataset
print(df.isnull().sum()) # Check for missing values in the dataset

X = df.drop("Y house price of unit area", axis=1)# Define the feature matrix X by dropping the target variable "Y house price of unit area" from the dataset. 
#This will be used for training the regression models.
y = df["Y house price of unit area"]# Define the target variable y as the "Y house price of unit area" column,
# which contains the house prices that we want to predict using the regression models.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Split the dataset into training and testing sets, with 80% of the data used for training and 20% for testing. 
# The random_state parameter ensures reproducibility of the results.

# Standardize the feature data to have a mean of 0 and a standard deviation of 1, which can improve the performance of many machine learning algorithms.
# The scaler is fitted on the training data and then applied to both the training and testing sets to ensure that they are on the same scale.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)# Fit the scaler to the training data and transform it, which standardizes the features in the training set.
X_test = scaler.transform(X_test)# Transform the testing data using the same scaler fitted on the training data, ensuring that the test features are standardized in the same way as the training features.

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    results[name] = rmse

    print(f"{name} RMSE: {rmse:.2f}")

    plt.figure(figsize=(8,6))
    plt.scatter(y_test, predictions, alpha=0.6)
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        'r--',
        linewidth=2
    )
    plt.xlabel("Actual House Price")
    plt.ylabel("Predicted House Price")
    plt.title(f"{name} - Actual vs Predicted")
    plt.show()

best_model = min(results, key=results.get)

print("\nModel Comparison")
for model, rmse in results.items():
    print(f"{model}: {rmse:.2f}")

print(f"\nBest Model: {best_model}")
print(f"Lowest RMSE: {results[best_model]:.2f}")
# The code reads a real estate dataset, preprocesses it, and evaluates multiple regression models to predict house prices. It calculates the RMSE for each model to compare their performance.
# Note: Ensure that the dataset "Real-estate1.csv" is in the same directory as this script or provide the correct path to it.
# Visual Representation of Predictions line is added to evaluate the performance of each model on the test set, and the RMSE is printed for comparison.
    
