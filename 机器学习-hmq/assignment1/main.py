from sklearn import datasets
from model import _k_fold, KNN, data_split, score
from show import show
import sys

if __name__ == '__main__':
    try:
        from sklearn import datasets
        iris = datasets.load_iris()
    except ImportError:
        with open('./data_path.txt') as path_file:
            filename = path_file.read()
        iris = dict()
        iris['data'] = []
        iris['target'] = []
        import csv
        data_reader = csv.reader(open(filename))
        tmp_str = ""
        count = -1
        for row in data_reader:
            if len(row) == 5:
                iris['data'].append([float(x) for x in row[:4]])
                if tmp_str == row[4]:
                    iris['target'].append(count)
                else:
                    count += 1
                    tmp_str = row[4]
                    iris['target'].append(count)
    xlabels = []
    results = []
    
    if len(sys.argv) <= 1 or str(sys.argv[1]) == 'simple':
        split_rates = [0.1, 0.2, 0.3, 0.5, 0.6]
        for rate in split_rates:
            xlabel = []
            result = []
            for i in range(25):
                knn = KNN(i+1)
                train_data, train_label, test_data, test_label = data_split(iris['data'], iris['target'], split_rate=rate)
                knn.get_data(train_data, train_label)
                label_result = [knn.predict(p) for p in test_data]
                xlabel.append(i+1)
                result.append(round(score(label_result, test_label), 4))
            xlabels.append(xlabel)
            results.append(result)
        show(xlabels, results, split_rates)
    elif str(sys.argv[1]) == 'k_fold':
        k = 10
        if len(sys.argv) >= 3:
            k = int(sys.argv[2])
        data_group_list, label_group_list = _k_fold(iris['data'], iris['target'])
        xlabel = []
        results = []
        for i in range(25):
            knn = KNN(i+1)
            score_list = []
            result = []
            for j in range(k):
                train_data = []
                train_label = []
                test_data = []
                test_label = []
                for index in range(k):
                    if index == j:
                        test_data[len(test_data): len(test_data)] = data_group_list[index]
                        test_label[len(test_label): len(test_label)] = label_group_list[index]
                    else:
                        train_data[len(train_data): len(train_data)] = data_group_list[index]
                        train_label[len(train_label): len(train_label)] = label_group_list[index]
                knn.get_data(train_data, train_label)
                label_result = [knn.predict(p) for p in test_data]
                result.append(round(score(label_result, test_label), 4))
            xlabel.append(i+1)
            results.append(round(sum(result)/k, 4))
        show(xlabel, results)
                
                    
