import numpy as np
import matplotlib.pyplot as plt
from analysis import result_analysis as rn

def create_confusion_matrix(actual, predicted):
    confusion_matrix = {}
    confusion_matrix['TP'] = np.sum((actual == 1) & (predicted == 1))
    confusion_matrix['TN'] = np.sum((actual == 0) & (predicted == 0))
    confusion_matrix['FP'] = np.sum((actual == 0) & (predicted == 1))
    confusion_matrix['FN'] = np.sum((actual == 1) & (predicted == 0))
    return confusion_matrix

def plot_confusion_matrix_and_table(svm_cm, logistic_cm):
    TP1 = svm_cm['TP']
    TN1 = svm_cm['TN']
    FP1 = svm_cm['FP']
    FN1 = svm_cm['FN']

    confusion_matrix1 = np.array([[TP1, FP1], [FN1, TN1]])

    # Plot the confusion matrix
    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
    fig.tight_layout(pad=5.0)
    fig.suptitle("Comparative analysis of SVM and Logistic Regression", fontsize=24)
    fig.patch.set_facecolor('xkcd:mint green')
    fig.canvas.manager.full_screen_toggle()

    im = ax0.imshow(confusion_matrix1, cmap='Set2')

    # Set axis labels
    ax0.set_title("Confusion Matrix of SVM")
    ax0.set_xticks([0, 1])
    ax0.set_yticks([0, 1])
    ax0.set_xticklabels(['Actual 1', 'Actual 0'])
    ax0.set_yticklabels(['Predicted 1', 'Predicted 0'])
    ax0.set_xlabel('Actual label')
    ax0.set_ylabel('Predicted label')

    # Add text annotations
    for i in range(2):
        for j in range(2):
            ax0.text(j, i, str(confusion_matrix1[i, j]), ha='center', va='center', color='white', fontsize="large")

    # Creating object of analysis class for analysis svm model
    svm_analysis = rn(svm_cm)

    # Plot tables
    ax2.set_title("Result Analysis Table")
    columns = ["Values"]
    rows = ["True Positive", "True Negative", "False Positive", "False Negative", "Accuracy", "Precision", "Recall", "F-score"]
    table_vals1 = [[TP1], [TN1], [FP1], [FN1], [svm_analysis.accuracy()], [svm_analysis.precision()], [svm_analysis.recall()], [svm_analysis.f_score()]]
    ax2.axis("off")
    table1 = ax2.table(cellText=table_vals1, rowLabels=rows, colLabels=columns, loc="upper center", colWidths=[1, 1])
    table1.auto_set_font_size(False)
    table1.set_fontsize(14)
    table1.auto_set_column_width(False)
    table1.scale(1, 2)

    TP2 = logistic_cm['TP']
    TN2 = logistic_cm['TN']
    FP2 = logistic_cm['FP']
    FN2 = logistic_cm['FN']

    confusion_matrix2 = np.array([[TP2, FP2], [FN2, TN2]])

    # Plot the confusion matrix
    im = ax1.imshow(confusion_matrix2, cmap='Set2')

    # Set axis labels
    ax1.set_title("Confusion Matrix of Logistic Regression")
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['Actual 1', 'Actual 0'])
    ax1.set_yticklabels(['Predicted 1', 'Predicted 0'])
    ax1.set_xlabel('Actual label')
    ax1.set_ylabel('Predicted label')

    # Add text annotations
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, str(confusion_matrix2[i, j]), ha='center', va='center', color='white', fontsize="large")

    # Creating object of analysis class for analysis logistic model
    logistic_analysis = rn(logistic_cm)

    # Add table
    ax3.set_title("Result Analysis Table")
    table_vals2 = [[TP2], [TN2], [FP2], [FN2], [logistic_analysis.accuracy()], [logistic_analysis.precision()], [logistic_analysis.recall()], [logistic_analysis.f_score()]]
    ax3.axis("off")
    table2 = ax3.table(cellText=table_vals2, rowLabels=rows, colLabels=columns, loc="upper center", colWidths=[1, 1])
    table2.auto_set_font_size(False)
    table2.set_fontsize(14)
    table2.auto_set_column_width(False)
    table2.scale(1, 2)

    # Show the plot
    plt.show()
