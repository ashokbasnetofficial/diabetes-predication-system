import numpy as np
import statistics

class KNN_Classifier():
    # initiating the parameters
    def __init__(self, distance_metric):
        self.distance_metric = distance_metric

    # getting the distance metric
    def get_distance_metric(self, training_data_point, test_data_point):
        if (self.distance_metric == 'euclidean'):
            dist = 0
            for i in range(len(training_data_point) - 1):
                dist = dist + (training_data_point[i] - test_data_point[i])**2
            euclidean_dist = np.sqrt(dist)
            return euclidean_dist

        elif (self.distance_metric == 'manhattan'):
            dist = 0
            for i in range(len(training_data_point) - 1):
                dist = dist + abs(training_data_point[i] - test_data_point[i])
            manhattan_dist = dist
            return manhattan_dist

    # getting the nearest neighbors
    def nearest_neighbors(self, X_train, test_data, k):
        distance_list = []
        for training_data in X_train:
            distance = self.get_distance_metric(training_data, test_data)
            distance_list.append((training_data, distance))
        distance_list.sort(key=lambda x: x[1])
        neighbors_list = []

        for j in range(k):
            neighbors_list.append(distance_list[j][0])
        return neighbors_list

    # predict the class of the new data point:
    def predict(self, X_train, test_data, k):
        neighbors = self.nearest_neighbors(X_train, test_data, k)
        for data in neighbors:
            label = []
            label.append(data[-1])
        predicted_class = statistics.mode(label)
        return predicted_class
