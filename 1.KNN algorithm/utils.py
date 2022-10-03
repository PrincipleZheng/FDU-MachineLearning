'''
Author: Zheng
Date: 2022-09-18 00:12:36
LastEditors: Zheng
LastEditTime: 2022-09-26 21:49:12
Description: default description
'''
import random
import matplotlib.pyplot as plt

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

def data_split(data, label, split_rate=0.2):
    """split the data according to the split rate

    Args:
        data (list): the raw dataset, it should be list or numpy.array
        label (type): the raw label dataset, it should be list or numpy.array
        split_rate (float, optional):  Defaults to 0.2.

    Returns:
        [train_data, test_data, train_label, test_label]
    """
    random.seed(2022)
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

def show(x_labels, y_labels, *arg):
    if len(arg) != 0:
        rates = arg[0]
        for i in range(len(x_labels)):
            x_label = x_labels[i]
            y_label = y_labels[i]
            rate = rates[i]
            plt.plot(x_label, y_label, label='split rate='+str(rate))
            plt.scatter(x_label, y_label)
        # for i in range(len(x_label)):
        #     if i % 2 == 0:
        #         plt.annotate(y_label[i], xy=(x_label[i], y_label[i]), xytext=(x_label[i], y_label[i]+0.001), weight='light')
        #     else:
        #         plt.annotate(y_label[i], xy=(x_label[i], y_label[i]), xytext=(x_label[i], y_label[i]-0.003), weight='light')
    else:
        plt.plot(x_labels, y_labels, label='KFold')
        plt.scatter(x_labels, y_labels)
    plt.legend(loc='best')
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.title('Accuracy of KNN algorithm with differenct k value')
    plt.show()
