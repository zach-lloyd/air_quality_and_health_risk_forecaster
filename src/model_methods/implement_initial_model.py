import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np

def implement_initial_model(df, y_feature):
    """
    Builds and runs an XGBoost model with the specified feature as the independent
    variable.
    
    :param df: The dataframe to analyze.
    :param y_feature: The feature from the dataframe that you want the model to 
    predict.

    returns: The Root Mean Squared Error and Mean Absolute Percentage Error of the
    model.
    """
    # Split the specified y feature from the rest of the data so it can be used
    # as the dependent variable.
    X = df.drop(y_feature, axis=1)
    y = df[y_feature]

    # Conduct the train-test split with standard parameters.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the model with default parameters. Tuning will occur in subsequent tasks.
    model = xgb.XGBRegressor(random_state=42)

    # Fit the model to the training data.
    model.fit(X_train, y_train)

    # Run the model on the test data.
    predictions = model.predict(X_test)

    # Calculate and return Root Mean Squared Error and Mean Absolute Percentage
    # Error, the two metrics that will be used to evaluate the model.
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mape = mean_absolute_percentage_error(y_test, predictions)

    return rmse, mape
