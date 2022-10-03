'''
Author: Zheng
Date: 2022-09-18 00:07:02
LastEditors: Zheng
LastEditTime: 2022-09-26 22:42:48
Description: default description
'''
from ctypes import pointer
import math
import numpy as np

class KNN(object):
    def __init__(self, k=4):
        """Initialize the model with the given k value

        Args:
            k (int, optional): Defaults to 4.

        Raises:
            ValueError: k value must be a positive number
        """
        if k <= 0:
            raise ValueError("Invalid k value!It should be a positive number")
        self.k = k
        self.data = None
        self.label = None

    def get_data(self, data, label):
        """This function is used to load the data

        Args:
            data (list): the feature of the dataset
            label (list): the label of the dataset
        """
        self.data = data
        self.label = label

    @staticmethod
    def get_euclidean_distance(point_1, point_2):
        p1 = np.array(point_1)
        p2 = np.array(point_2)
        return np.sqrt(np.sum(np.square(p1 - p2)))
    
    @staticmethod
    def get_manhattan_distance(point_1, point_2):
        p1 = np.array(point_1)
        p2 = np.array(point_2)
        return np.linalg.norm(p1 - p2, ord=1)
    
    
    def predict(self, point):
        """prediction the most probable label

        Args:
            point (list): the point to predict

        Returns:
            int : the label of the prediction
        """
        distance = dict()
        for i in range(len(self.data)):
            distance[i] = self.get_manhattan_distance(self.data[i], point)
            # distance[i] = self.get_euclidean_distance(self.data[i], point)
        sorted_index = [item[0] for item in sorted(distance.items(), key=lambda x: x[1])]
        top_k_index = sorted_index[:self.k]
        top_k = dict()
        label_set = set(self.label)
        for i in label_set:
            top_k[i] = 0
        for p in top_k_index:
            top_k[self.label[p]] += 1                             # before the optimization
            # top_k[self.label[p]] += 100 / (distance[p] + 1)     # adjust the weight for the neighbor according to the distance
        top_k = sorted(top_k.items(), key=lambda x: -x[1])
        return top_k[0][0]
        