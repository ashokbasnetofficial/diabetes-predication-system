import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_summary_data():
    # Read analysis data from csv
    svm_data = pd.read_csv("./summary_data/svm.csv")
    logistic_data = pd.read_csv("./summary_data/logistic.csv")
    
    # Calculate mean of each analysis parameter
    svm_data_mean = svm_data.mean()
    logistic_data_mean = logistic_data.mean()
    
    # Create array of means of each model
    svm_parameter_mean = [svm_data_mean["Accuracy"], svm_data_mean["Precision"], svm_data_mean["Recall"], svm_data_mean["F-score"]]
    logistic_parameter_mean = [logistic_data_mean["Accuracy"], logistic_data_mean["Precision"], logistic_data_mean["Recall"], logistic_data_mean["F-score"]]
    
    # Create figures and subplots
    fig, ax0 = plt.subplots(figsize=(8, 10))
    fig.tight_layout(pad=8.0)
    fig.suptitle("Data Summary", fontsize=24)
    fig.patch.set_facecolor('xkcd:mint green')
    fig.subplots_adjust(bottom=0.09)
    
    # Create array of labels for x axis
    X_name = ["Accuracy", "Precision", "Recall", "F-Score"]
    X = np.arange(len(X_name))
    
    # Configure first subplot to represent mean
    max_y_value = max(max(svm_parameter_mean), max(logistic_parameter_mean))
    ax0.set_ylim([0, max_y_value + 10])
    ax0.bar(X + 0.00, svm_parameter_mean, color='b', width=0.35, label="SVM")
    ax0.bar(X + 0.35, logistic_parameter_mean, color='r', width=0.35, label="Logistic Regression")
    ax0.set_xticks(X + 0.175)
    ax0.set_xticklabels(X_name)
    ax0.set_xlabel("Analysis Parameter")
    ax0.set_ylabel("Analysis Parameter Value")
    ax0.set_title("Mean of Analysis Parameter")
    ax0.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax0.grid(True, axis="y", linestyle='--', linewidth=0.5)
    
    # Set y-tick labels with interval of 5
    y_tick_labels = np.arange(0, max_y_value + 10, 5)
    ax0.set_yticks(y_tick_labels)
    ax0.set_yticklabels(y_tick_labels)
    
    # Display the value of each bar at the top of the bar
    for i, v in enumerate(svm_parameter_mean):
        ax0.text(i - 0.12, v + 0.01, str(round(v, 2)), color='blue', fontweight='bold')
    for i, v in enumerate(logistic_parameter_mean):
        ax0.text(i + 0.23, v + 0.01, str(round(v, 2)), color='red', fontweight='bold')
    
    plt.show()
