'''
Author: Zheng
Date: 2022-09-17 21:37:18
LastEditors: Zheng
LastEditTime: 2022-09-26 22:44:35
Description: default description
'''
import csv
import sys
from KNN import KNN
from utils import *
from tqdm import tqdm

if __name__ == '__main__':
    filename = '1.KNN algorithm/whitewine.csv'
    data_reader = csv.reader(open(filename))
    header = next(data_reader)
    winequality = dict()
    winequality['data'] = []
    winequality['target'] = []
    for row in data_reader:
        dataArray = []
        # TODO：处理数据缺失的情况 暂时用0代替
        for x in row[1:-1]:
            if x == '':
                dataArray.append(float(0))
            else:
                dataArray.append(float(x))
        winequality['data'].append(dataArray)
        # print(winequality['data'][1])
        winequality['target'].append(row[-1])
        # print(winequality['target'])
    xlabels = []
    results = []
    if len(sys.argv) <= 1 or str(sys.argv[1]) == 'simple':
        split_rates = [0.1,0.2,0.3,0.4,0.5]
        for rate in tqdm(split_rates):
            xlabel = []
            result = []
            for i in tqdm(range(2,10)):
                knn = KNN(i+1)
                train_data, train_label, test_data, test_label = data_split(winequality['data'], winequality['target'], split_rate=rate)
                knn.get_data(train_data, train_label)
                label_result = [knn.predict(p) for p in test_data]
                xlabel.append(i+1)
                result.append(round(score(label_result, test_label), 4))
            xlabels.append(xlabel)
            results.append(result)
        show(xlabels, results, split_rates)




