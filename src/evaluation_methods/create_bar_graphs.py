import matplotlib.pyplot as plt
import numpy as np

def create_bar_graphs(baseline_rmses, baseline_mapes, optimized_rmses, optimized_mapes):
    """
    Create figure with two bar graphs, one comparing baseline and optimized model RMSEs
    and one comparing baseline and optimized model MAPEs.
    
    :param baseline_rmses: RMSE values for the baseline air quality and health 
    risk models.
    :param baseline_mapes: MAPE values for the baseline air quality and health
    risk models.
    :param optimized_rmses: RMSE values for the optimized air quality and health 
    risk models.
    :param optimized_mapes: MAPE values for the optimized air quality and health
    risk models.
    """
    labels = ["Air Quality", "Health Risk"]

    x = np.arange(len(labels))  
    width = 0.35  

    # Create the figure (1 Row, 2 Columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 6))

    # Plot RMSE
    rects1 = ax1.bar(x - width/2, baseline_rmses, width, label = "Baseline", color = "#4c72b0")
    rects2 = ax1.bar(x + width/2, optimized_rmses, width, label = "Optimized", color = "#55a868")

    ax1.set_ylabel("RMSE Score")
    ax1.set_title("RMSE Comparison\n(Lower is Better)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    ax1.grid(axis = "y", linestyle = "--", alpha = 0.7)

    # Add labels on top of bars
    ax1.bar_label(rects1, padding = 3, fmt = "%.5f")
    ax1.bar_label(rects2, padding = 3, fmt = "%.5f")

    # Plot MAPE
    rects3 = ax2.bar(x - width/2, baseline_mapes, width, label = "Baseline", color = "#4c72b0")
    rects4 = ax2.bar(x + width/2, optimized_mapes, width, label = "Optimized", color = "#55a868")

    ax2.set_ylabel("MAPE (Decimal)")
    ax2.set_title("MAPE Comparison\n(Lower is Better)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()
    ax2.grid(axis = "y", linestyle = "--", alpha = 0.7)

    # Add labels on top of bars
    ax2.bar_label(rects3, padding = 3, fmt = "%.5f")
    ax2.bar_label(rects4, padding = 3, fmt = "%.5f")

    # Save chart to file
    plt.tight_layout()
    plt.savefig("../visualizations/model_comparison_chart.png")
    print("Chart saved as 'model_comparison_chart.png'")
