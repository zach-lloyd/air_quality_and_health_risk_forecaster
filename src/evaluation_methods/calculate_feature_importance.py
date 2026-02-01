import matplotlib.pyplot as plt
import xgboost as xgb

def calculate_feature_importance(model, feature_name):
    """
    Plots bar chart of the feature importances of the given model.
    
    :param model: An XGBoost model.
    :param feature_name: The name of the dependent feature (used in the chart's
    title).
    """
    plt.figure(figsize = (10, 6))
    xgb.plot_importance(
        model,
        #max_num_features = 10, 
        importance_type = "weight", 
        title = f"{feature_name} Feature Importance"
    )
    plt.tight_layout()
    plt.savefig(f"{feature_name} Feature Importance.png")
    print(f"{feature_name} Feature Importance Chart Saved")
