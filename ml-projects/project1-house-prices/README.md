# This code reads a real estate dataset, preprocesses it, and evaluates multiple regression models to predict house prices using the models which are specified. 
# It calculates the RMSE for each model to compare their performance.
# Note: Ensure that the dataset "Real-estate1.csv" is in the same directory as this script or provide the correct path to it.
# Visual Representation of Predictions line is added to evaluate the performance of each model on the test set, and the RMSE is printed for comparison.
# The code uses pandas for data manipulation, scikit-learn for machine learning models and evaluation, and matplotlib for visualization.
# The best model is determined based on the lowest RMSE value, which indicates better performance in predicting house prices.
# The models evaluated include Linear Regression, Decision Tree, Random Forest, and Gradient Boosting, which are common regression algorithms used in machine learning for predictive modeling.