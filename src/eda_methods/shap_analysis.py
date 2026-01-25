import matplotlib.pyplot as plt
import shap

def shap_analysis(model, x_test, feature_name):
    """
    Calculates and plots Shapley values of the given model on the test data.
    
    :param model: An XGBoost model.
    :param x_test: The independent feature test data.
    :param feature_name: The name of the dependent feature (used in the chart's
    title).
    """
    # Initialize the SHAP explainer
    explainer = shap.Explainer(model)
    shap_values = explainer(x_test)

    # Plot A: SHAP Summary Plot
    plt.figure(figsize = (12, 8))
    plt.title(f"SHAP Summary Plot: {feature_name}", fontsize=16)
    shap.summary_plot(shap_values, x_test, max_display = 30, show = False)
    plt.tight_layout()
    plt.savefig(f"{feature_name} Shap Summary.png")

    # Plot B: SHAP Bar Plot
    plt.figure(figsize = (10, 6))
    plt.title(f"Mean |SHAP| Value: {feature_name}", fontsize = 16)
    shap.plots.bar(shap_values, max_display = 30, show = False)
    plt.tight_layout()
    plt.savefig(f"{feature_name} Shap Analysis.png")
    print(f"{feature_name} Shap Analysis Chart Saved")
