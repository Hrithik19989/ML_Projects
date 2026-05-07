import pandas as pd # pandas is a powerful library in Python used for data manipulation and analysis. It provides data structures like DataFrames that allow for easy handling of structured data, making it ideal for tasks such as data cleaning, transformation, and analysis.
from sklearn.model_selection import train_test_split # 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("Real-estate1.csv")
df = df.select_dtypes(include=[np.number]).dropna()
print(df.head())
print(df.describe())
print(df.isnull().sum())

X = df.drop("Y house price of unit area", axis=1)
y = df["Y house price of unit area"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"{name} → RMSE: {rmse:.2f}")
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, preds, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'b--')
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title(f'{name} Predictions vs Actual')
    plt.show()
    
# Best model is determined based on the lowest RMSE value, which indicates better performance in predicting house prices.
best_model = min(models, key=lambda name: np.sqrt(mean_squared_error(y_test, models[name].predict(X_test))))
print(f"Best Model: {best_model}")
    
# The code reads a real estate dataset, preprocesses it, and evaluates multiple regression models to predict house prices. It calculates the RMSE for each model to compare their performance.
# Note: Ensure that the dataset "Real-estate1.csv" is in the same directory as this script or provide the correct path to it.
# Visual Representation of Predictions line is added to evaluate the performance of each model on the test set, and the RMSE is printed for comparison.
    
