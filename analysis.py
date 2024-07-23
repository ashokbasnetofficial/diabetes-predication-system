class result_analysis:
    
    def __init__(self,confusion_matrix_dict):
        self.confusion_matrix_dict = confusion_matrix_dict
    
    def accuracy(self):
        total_predicitions = sum(self.confusion_matrix_dict.values())
        return round(((self.confusion_matrix_dict["TP"]+self.confusion_matrix_dict["TN"])/total_predicitions)*100,3)
    
    def precision(self):
        return round((self.confusion_matrix_dict["TP"]/(self.confusion_matrix_dict["FP"]+self.confusion_matrix_dict["TP"]))*100,3)

    def recall(self):
        return round((self.confusion_matrix_dict["TP"]/(self.confusion_matrix_dict["TP"]+self.confusion_matrix_dict["FN"]))*100,3)
    
    def f_score(self):
        precision_times_recall = self.precision() * self.recall()
        precision_sum_recall = self.precision() + self.recall()
        return round((2*(precision_times_recall/precision_sum_recall)),3)

