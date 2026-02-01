import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np

# Run a grid search on the model using the various parameters specified above to 
# determine the most optimal model
def optimize_model(X_train, y_train, X_test, y_test, param_grid):
    """
    Creates an XGBoost model using the provided train/test split, then runs
    a grid search to find the best parameters from the parameter grid.
    
    :param X_train: Independent variables training data.
    :param y_train: Dependent variable training data.
    :param X_test: Indepdent variables test data.
    :param y_test: Dependent variable test data.
    :param param_grid: Dictionary containing the parameters to test during the
    grid search.

    Returns: The best model, a grid of the best parameters, the best RMSE, and 
    the best MAPE.
    """
    # n_jobs = -1 uses all processors for speed
    xgb_model = xgb.XGBRegressor(random_state = 42, n_jobs = -1)

    grid_search = GridSearchCV(
        estimator = xgb_model,
        param_grid = param_grid,
        # Use RMSE for scoring because it aligns with my previous use of RMSE as 
        # one of the metrics to evaluate the model. However, negative RMSE needs
        # to be used here to ensure the model minimizes it instead of maximizing
        # it.
        scoring = "neg_root_mean_squared_error",
        cv = 3, # 3-fold Cross Validation
        verbose = 1
    )

    print("Starting Grid Search...")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best Parameters Found: {grid_search.best_params_}")

    # After picking the best model, evaluate it on the test data. For a fair
    # comparison, use the same metrics used in Task 1 (RMSE and MAPE) 
    predictions = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mape = mean_absolute_percentage_error(y_test, predictions)

    # Return the best model, evaluation metrics, and the parameters used to construct
    # the best model
    return best_model, rmse, mape, grid_search.best_params_
