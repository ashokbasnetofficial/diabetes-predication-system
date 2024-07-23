from matplotlib.pyplot import axis
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from SVM import SVM_classifier as svm
from Logistic_regression import Logistic_Regression as lr
from draw import create_confusion_matrix, plot_confusion_matrix_and_table
from analysis_summary_csvgen import analysis_summary_csvgen
from analysis import result_analysis as rn

def run_calculation(file_path):
    # Reading dataset
    dataset = pd.read_csv(file_path)

    # Selecting outcome from dataset
    target = dataset['Outcome']

    # Standardization features of dataset or data preprocessing
    features = dataset.drop(columns="Outcome", axis=1)
    scaler = StandardScaler()
    std_data = scaler.fit_transform(features)  # Normalize the features with mean of 0 and standard deviation of 1
    features = std_data

    # Splitting features into training dataset and testing dataset
    X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.2, random_state=2, stratify=target)

    # Creating object for SVM model training and testing
    svm_classifier = svm(learning_rate=0.001, no_of_iterations=1000, lambda_parameter=0.01)
    svm_classifier.fit(X_train, Y_train)
    # Predicting the test features
    svm_X_test_prediction = svm_classifier.predict(X_test)

    # Creating object for logistic regression model training and testing
    logistic_classifier = lr(learning_rate=0.001, no_of_iterations=1000)
    logistic_classifier.fit(X_train, Y_train)
    # Predicting the test features
    logistic_X_test_prediction = logistic_classifier.predict(X_test)

    # Creating confusion matrix by comparing actual outcome and predicted outcome for SVM model
    svm_cm = create_confusion_matrix(actual=Y_test, predicted=svm_X_test_prediction)

    # Creating confusion matrix by comparing actual outcome and predicted outcome for logistic regression model
    logistic_cm = create_confusion_matrix(actual=Y_test, predicted=logistic_X_test_prediction)

    # Plotting confusion matrix and table for SVM and logistic regression
    plot_confusion_matrix_and_table(svm_cm=svm_cm, logistic_cm=logistic_cm)

    # Saving analysis parameter of each model to its own file
    csvgen = analysis_summary_csvgen(svm_cm, logistic_cm)
    csvgen.svm_csv()
    csvgen.logistic_csv()
