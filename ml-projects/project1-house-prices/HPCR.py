import pandas as pd
from sklearn.model_selection import train_test_split # 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# This code reads a real estate dataset, preprocesses it, and evaluates multiple regression models to predict house prices using the models which are specified. 
# It calculates the RMSE for each model to compare their performance.
# Note: Ensure that the dataset "Real-estate1.csv" is in the same directory as this script or provide the correct path to it.
# Visual Representation of Predictions line is added to evaluate the performance of each model on the test set, and the RMSE is printed for comparison.
# The code uses pandas for data manipulation, scikit-learn for machine learning models and evaluation, and matplotlib for visualization.
# The best model is determined based on the lowest RMSE value, which indicates better performance in predicting house prices.
# The models evaluated include Linear Regression, Decision Tree, Random Forest, and Gradient Boosting, which are common regression algorithms used in machine learning for predictive modeling.

# Load the dataset from a CSV file named "Real-estate1.csv" using pandas.
df = pd.read_csv("Real-estate1.csv")
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
    "Linear Regression": LinearRegression(),# Define a dictionary of regression models to evaluate, including Linear Regression, Decision Tree, Random Forest, and Gradient Boosting.
    "Decision Tree": DecisionTreeRegressor(),# Each model is instantiated and stored in the dictionary with a descriptive name as the key.
    "Random Forest": RandomForestRegressor(),# The models will be trained and evaluated in a loop, where the RMSE for each model will be calculated and printed,
    #along with a scatter plot of predicted vs actual prices.
    "Gradient Boosting": GradientBoostingRegressor()# The best model will be determined based on the lowest RMSE value,
    #which indicates better performance in predicting house prices.
}

for name, model in models.items():
    model.fit(X_train, y_train)# Fit each model to the training data, allowing it to learn the relationships between the features and the target variable (house prices).
    preds = model.predict(X_test)# Use the trained model to make predictions on the test set, which will be compared to the actual house prices to evaluate the model's performance.
    rmse = np.sqrt(mean_squared_error(y_test, preds))# Calculate the Root Mean Squared Error (RMSE) for the model's predictions, which is a common metric for evaluating regression models.
    print(f"{name} → RMSE: {rmse:.2f}")
    plt.figure(figsize=(10, 6))# Create a scatter plot to visualize the relationship between the actual house prices (y_test) and the predicted prices (preds) for each model.
    plt.scatter(y_test, preds, alpha=0.5)# The scatter plot will show how well the predicted prices align with the actual prices, where points closer to the diagonal line (y = x) indicate better predictions.
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'b--')# Add a dashed blue line representing the ideal scenario where predicted prices perfectly match actual prices (y = x).
    plt.xlabel('Actual Prices')# Set the x-axis label to "Actual Prices" to indicate that the horizontal axis represents the true house prices from the test set.
    plt.ylabel('Predicted Prices')# Set the y-axis label to "Predicted Prices" to indicate that the vertical axis represents the house prices predicted by the model.
    plt.title(f'{name} Predictions vs Actual')# Set the title of the plot to indicate which model's predictions are being visualized, 
    #allowing for a clear comparison of the predicted vs actual prices for each regression model.
    plt.show()# Display the scatter plot for each model, allowing for a visual comparison of the predicted vs actual prices.
    
# Best model is determined based on the lowest RMSE value, which indicates better performance in predicting house prices.
best_model = min(models, key=lambda name: np.sqrt(mean_squared_error(y_test, models[name].predict(X_test))))
print(f"Best Model: {best_model}")
    
# The code reads a real estate dataset, preprocesses it, and evaluates multiple regression models to predict house prices. It calculates the RMSE for each model to compare their performance.
# Note: Ensure that the dataset "Real-estate1.csv" is in the same directory as this script or provide the correct path to it.
# Visual Representation of Predictions line is added to evaluate the performance of each model on the test set, and the RMSE is printed for comparison.
    
