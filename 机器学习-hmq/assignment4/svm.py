from sklearn import datasets
import numpy as np


class SVM:
    def __init__(self, C=10, degree=3, gamma=None, coef0=0.0, tol=0.001,
                 lr=0.01, max_iter=10000):
        self.C = C
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.lr = lr
        self.max_iter = max_iter
        self.alpha = None
        self.w = None
        self.b = None

    # 定义多项式核函数
    @staticmethod
    def _poly(x, y, p=4):
        return (x.dot(y.T)+1)**p

    # 定义RBF核函数
    @staticmethod
    def _rbf(x, y, gamma):
        return np.exp(-gamma * np.sum((x[..., None, :] - y)**2, axis=2))

    # 不适用核函数
    @staticmethod
    def _origin(x, _):
        return x

    def __E(self, i):
        return self.b + self.y[i] * self.alpha.dot(self.X[i])

    def __selectJ(i, m):
        j = i
        while j == i:
            j = int(np.random.uniform(0, m))
        return j
    
    def __clipAlpha(aj, H, L):
        if aj > H:
            aj = H
        if L > aj:
            aj = L
        return aj
    
    # 小范围随机梯度下降
    def __gradientDescent(self, X, y, kernel='poly', p=4, batch_size=100):
        if kernel == 'poly':
            self.kernel = lambda _x, _y: self._poly(_x, _y, p)
        elif kernel == 'rbf':
            self.gamma = 1 / X.shape[1] if self.gamma is None else self.gamma 
            self.kernel = lambda _x, _y: self._rbf(_x, _y, self.gamma)
        elif kernel == 'origin':
            self.kernel = lambda _x, _: self._origin(_x, _)
        else:
            raise NotImplementedError('kernel {} not implemented'.format(kernel))
        self.alpha = np.zeros(X.shape[0])
        self.b = 0.0
        kMat = self.kernel(X, X)
        print('kMat shape: {}'.format(kMat.shape))
        for _ in range(self.max_iter):
            indice = np.random.permutation(len(y))[:batch_size]
            k_mat_batch, y_batch = kMat[indice], y[indice]
            err = - y_batch * (k_mat_batch.dot(self.alpha) + self.b)    # TODO
            if np.max(err) < 0:
                continue
            mask = err >= 0
            delta = self.lr * y_batch[mask]
            self.alpha += np.sum(delta[..., None]*k_mat_batch[mask], axis=0)
            self.b += np.sum(delta)
        return self.b, self.alpha

    def fit(self, X, y, method='smo', kernel='poly'):
        self.X = X
        self.y = y
        if method == 'smo':
            self.b, self.alpha = self.__smoSimple(X, y)
        elif method == 'gradient':
            self.b, self.alpha = self.__gradientDescent(X, y, kernel)

    def __smoSimple(self, X, y):
        m, n = X.shape
        self.X = X
        self.y = y
        self.alpha = np.zeros(n)
        self.b = 0
        alphas = np.zeros(m)
        E = np.zeros(m)
        for i in range(m):
            E[i] = self.__E(i)
        iter = 0
        while iter < self.max_iter:
            alphaPairsChanged = 0
            for i in range(m):
                E[i] = self.__E(i)
                if (y[i] * E[i] < -self.tol and alphas[i] < self.C) or \
                        (y[i] * E[i] > self.tol and alphas[i] > 0):
                    j = self.__selectJ(i, m)
                    E[j] = self.__E(j)
                    alphaIold = alphas[i].copy()
                    alphaJold = alphas[j].copy()
                    if y[i] != y[j]:
                        L = max(0, alphas[j] - alphas[i])
                        H = min(self.C, self.C + alphas[j] - alphas[i])
                    else:
                        L = max(0, alphas[j] + alphas[i] - self.C)
                        H = min(self.C, alphas[j] + alphas[i])
                    if L == H:
                        continue
                    eta = 2.0 * self.X[i].dot(self.X[j]) - self.X[i].dot(self.X[i]) - self.X[j].dot(self.X[j])
                    if eta >= 0:
                        continue
                    alphas[j] -= y[j] * (E[i] - E[j]) / eta
                    alphas[j] = self.__clipAlpha(alphas[j], H, L)
                    if abs(alphas[j] - alphaJold) < 0.00001:
                        continue
                    alphas[i] += y[j] * y[i] * (alphaJold - alphas[j])
                    b1 = self.b - E[i] - y[i] * self.X[i].dot(self.X[i]) * (alphas[i] - alphaIold) - y[j] * self.X[i].dot(self.X[j]) * (alphas[j] - alphaJold)
                    b2 = self.b - E[j] - y[i] * self.X[i].dot(self.X[j]) * (alphas[i] - alphaIold) - y[j] * self.X[j].dot(self.X[j]) * (alphas[j] - alphaJold)
                    if 0 < alphas[i] and alphas[i] < self.C:
                        self.b = b1
                    elif 0 < alphas[j] and alphas[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2.0
                    alphaPairsChanged += 1
                    print('iter: %d i: %d, pairs changed %d' % (iter, i, alphaPairsChanged))
            if alphaPairsChanged == 0:
                iter += 1
            else:
                iter = 0
        return self.b, self.alpha

    def predict(self, X, raw=False):
        # k_mat = self.kernel(self.X, X)
        y_pred = self.alpha.dot(X.T) + self.b
        if raw:
            return y_pred
        else:
            return np.sign(y_pred)

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
    print('data_size:', len(X))
    print('feature_numbers:', len(X[0]))
    print('target_names:',cancer.target_names)
    print(cancer.DESCR)
    print(X[0])
    from sklearn.preprocessing import MinMaxScaler
    X = MinMaxScaler().fit_transform(X=X)
    print(X[0]) 
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2021, shuffle=True)
    print('数据集size:\n', '训练集', X_train.shape, y_train.shape, '\n 测试集', X_test.shape, y_test.shape)
    svm = SVM()
    svm.fit(X_train, y_train, method='smo', kernel='poly')
    y_pred = svm.predict(X_test)
    print('训练集准确率:', score(y_pred, y_test))


