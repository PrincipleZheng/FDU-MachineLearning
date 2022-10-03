
import numpy as np
from sklearn import datasets
from Util import *

class KP:
    def __init__(self):
        self._x = None
        self._alpha = self._b = self._kernel = None
    
    # 定义多项式核函数
    @staticmethod
    def _poly(x, y, p=8):
        return (x.dot(y.T) + 1) ** p
    
    # 定义 rbf 核函数
    @staticmethod
    def _rbf(x, y, gamma):
        return np.exp(-gamma * np.sum((x[..., None, :] - y) ** 2, axis=2))
        
    def fit(self, x, y, kernel="poly", p=None, gamma=None, lr=0.001, batch_size=128, epoch=10000):
        x, y = np.asarray(x, np.float32), np.asarray(y, np.float32)
        if kernel == "poly":
            p = 4 if p is None else p
            self._kernel = lambda x_, y_: self._poly(x_, y_, p)
        elif kernel == "rbf":
            gamma = 1 / x.shape[1] if gamma is None else gamma
            self._kernel = lambda x_, y_: self._rbf(x_, y_, gamma)
        else:
            raise NotImplementedError("Kernel '{}' has not defined".format(kernel))
        self._alpha = np.zeros(len(x))
        self._b = 0.
        self._x = x
        k_mat = self._kernel(x, x)
        print('k_mat shape', k_mat.shape)
        for _ in range(epoch):
            indices = np.random.permutation(len(y))[:batch_size]
            k_mat_batch, y_batch = k_mat[indices], y[indices]
            err = -y_batch * (k_mat_batch.dot(self._alpha) + self._b)
            if np.max(err) < 0:
                continue
            mask = err >= 0
            delta = lr * y_batch[mask]
            self._alpha += np.sum(delta[..., None] * k_mat_batch[mask], axis=0)
            self._b += np.sum(delta)
    
    def predict(self, x, raw=False):
        x = np.atleast_2d(x).astype(np.float32)
        k_mat = self._kernel(self._x, x)
        y_pred = self._alpha.dot(k_mat) + self._b
        if raw:
            return y_pred
        return np.sign(y_pred).astype(np.float32)

def score(test_result, y_test):
    counter = 0
    acc = 0
    for (pred_label, true_label) in zip(test_result, y_test):
        counter += 1
        if pred_label == true_label:
            acc += 1
    return acc/counter

if __name__ == '__main__':
    cancer = datasets.load_breast_cancer()
    X = cancer.data
    y = cancer.target
    # print('data_size:', len(X))
    # print('feature_numbers:', len(X[0]))
    # print('target_names:',cancer.target_names)
    # print(cancer.DESCR)
    print(X[0])
    from sklearn.preprocessing import MinMaxScaler
    X = MinMaxScaler().fit_transform(X=X)
    print(X[0]) 
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2021, shuffle=True)
    print('数据集size:\n', '训练集', X_train.shape, y_train.shape, '\n 测试集', X_test.shape, y_test.shape)
    svm = KP()
    svm.fit(X_train, y_train, kernel='rbf')
    y_pred = svm.predict(X_test)
    print('训练集准确率:', score(y_pred, y_test))
    xs, ys = gen_spiral()
    svm.fit(xs, ys, kernel='poly')
    print("准确率：{:8.6} %".format((svm.predict(xs) == ys).mean() * 100))
    visualize2d(svm, xs, ys, True)
