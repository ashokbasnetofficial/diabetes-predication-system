import pandas as pd
import os
from analysis import result_analysis as rn

# Class for creating file for each algorithm to store analysis parameter
class analysis_summary_csvgen():
    def __init__(self, cm_svm, cm_logis):
        self.cm_svm = cm_svm
        self.cm_logis = cm_logis
        if not os.path.exists("./summary_data"):  # Create folder if not exist
            os.mkdir("./summary_data")
        
    # Return dictionary of calculated data
    def data_holder(self, analysis):
        data = {
            "Accuracy": analysis.accuracy(),
            "Precision": analysis.precision(),
            "Recall": analysis.recall(),
            "F-score": analysis.f_score()
        }
        return data

    # Create svm.csv file and write value received from data_holder
    def svm_csv(self):
        svm_analysis = rn(self.cm_svm)
        data = self.data_holder(svm_analysis)
        svm_df = pd.DataFrame(data=data, index=[0])
        svm_df.to_csv("./summary_data/svm.csv", mode="a", header=not os.path.exists("./summary_data/svm.csv"), index=False)
    
    # Create logistic.csv file and write value received from data_holder
    def logistic_csv(self):
        logistic_analysis = rn(self.cm_logis)
        data = self.data_holder(logistic_analysis)
        logistic_df = pd.DataFrame(data=data, index=[0])
        logistic_df.to_csv("./summary_data/logistic.csv", mode="a", header=not os.path.exists("./summary_data/logistic.csv"), index=False)
