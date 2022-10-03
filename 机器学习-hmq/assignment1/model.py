import math
import random
import copy
import numpy

def data_split(data, label, split_rate=0.2):
    """split the data according to the split rate

    Args:
        data (list): the raw dataset, it should be list or numpy.array
        label (type): the raw label dataset, it should be list or numpy.array
        split_rate (float, optional):  Defaults to 0.2.

    Returns:
        [train_data, test_data, train_label, test_label]
    """
    random.seed(2021)
    random.random()
    train_data, test_data, train_label, test_label = [[], [], [], []]
    for i in range(len(data)):
        if random.random() < split_rate:
            test_data.append(data[i])
            test_label.append(label[i])
        else:
            train_data.append(data[i])
            train_label.append(label[i])
    return train_data, train_label, test_data, test_label

def _k_fold(data, label, k=10):
    """k fold algorithm to split the data

    Args:
        data (list): the raw dataset, it should be list or list-like(like numpy.array or torch.tensor)
        label (type): the raw label dataset, it should be list or list-like(like numpy.array or torch.tensor)
        k (int, optional): the total number of the groups. Defaults to 10.

    Returns:
        data_group_list, label_group_list : the list of the groups
    """
    if k <= 0:
        raise ValueError("Invalid k value!It should be a positive number")
    data_group_list = []                # to store the divided data groups
    label_group_list = []               # to store the divided label groups
    group_count = len(data)//k          # the number of data in a single group
    if type(data) == numpy.ndarray:
        data = data.tolist()            # turn the numpy.array into list
        label = label.tolist()          # otherwise the shuffle may go wrong
    data_copy = copy.deepcopy(data)
    label_copy = copy.deepcopy(label)
    random.seed(2021)
    random.shuffle(data_copy)
    random.seed(2021)
    random.shuffle(label_copy)
    for i in range(k):
        if i != k - 1:
            data_group = data_copy[i*group_count : (i+1)*group_count]
            label_group = label_copy[i*group_count : (i+1)*group_count]
        else:
            data_group = data_copy[i*group_count:]
            label_group = label_copy[i*group_count:]
        data_group_list.append(data_group)
        label_group_list.append(label_group)
    return data_group_list, label_group_list

    
def score(pred, labels):
    """count the accuracy based on the test dataset

    Args:
        pred (list): the prediction of the test dataset label
        labels (list): the true values of the test dataset label

    Returns:
        score (float): the score of the prediction
    """
    count = 0
    for i in range(len(pred)):
        if pred[i] == labels[i]:
            count += 1
    score = count / len(pred)
    return score

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
    def get_distance(point_1, point_2):
        differ_pow2 = [(point_1[i] - point_2[i]) ** 2 for i in range(len(point_1))]
        return math.sqrt(sum(differ_pow2))
    
    def predict(self, point):
        """prediction the most probable label

        Args:
            point (list): the point to predict

        Returns:
            int : the label of the prediction
        """
        distance = dict()
        for i in range(len(self.data)):
            distance[i] = self.get_distance(self.data[i], point)
        sorted_index = [item[0] for item in sorted(distance.items(), key=lambda x: x[1])]
        top_k_index = sorted_index[:self.k]
        top_k = dict()
        label_set = set(self.label)
        for i in label_set:
            top_k[i] = 0
        for p in top_k_index:
            # top_k[self.label[p]] += 1                             # before the optimization
            top_k[self.label[p]] += 100 / (distance[p] + 1)         # adjust the weight for the neighbor according to the distance
        top_k = sorted(top_k.items(), key=lambda x: -x[1])
        return top_k[0][0]
        