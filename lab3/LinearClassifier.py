import numpy as np


def get_tony_dataset():
    tony_data = np.array([[1, 1], [2, 2], [2, 0],
                          [0, 0], [1, 0], [0, 1]])
    tony_label = np.array([0, 0, 0, 1, 1, 1])
    return tony_data, tony_label


def get_MNIST_dataset():
    train_file, train_labels_file = "data/TrainSamples.csv", "data/TrainLabels.csv"
    train_x = np.genfromtxt(train_file, delimiter=',')
    train_y = np.genfromtxt(train_labels_file, delimiter=',', dtype=np.int8)

    test_file, test_labels_file = "data/TestSamples.csv", "data/TestLabels.csv"
    test_x = np.genfromtxt(test_file, delimiter=',')
    test_y = np.genfromtxt(test_labels_file, delimiter=',')
    return (train_x, train_y), (test_x, test_y)


def preprocess_dataset(data, label, standardize=True):
    data = np.insert(data, 0, 1, axis=1)
    if standardize:
        for idx, l in enumerate(label):
            if l:
                data[idx] = -data[idx]
    return data


class Perceptron:
    def __init__(self, dim):
        self.w = np.random.randn(dim)

    def judge(self, data):
        for x in data:
            if np.dot(self.w, x) <= 0:
                return False
        return True

    def fit(self, data, label):
        data = preprocess_dataset(data, label)
        num, dim = data.shape
        k = 0
        while True:
            if np.dot(self.w, data[k]) <= 0:
                self.w += data[k]
            k = (k + 1) % num
            if self.judge(data):
                break


class LMSE:
    def __init__(self):
        self.w = None

    def fit(self, data, label):
        data = preprocess_dataset(data, label)
        num, dim = data.shape
        self.w = np.linalg.inv(data.T.dot(data)).dot(data.T).dot(np.ones((num, 1))).reshape((dim, ))


class KeslersPerceptron:
    def __init__(self, cls, dim, max_epoch=20):
        self.a = []
        self.cls = cls
        self.__max_epoch = max_epoch
        for i in range(cls):
            self.a.append(np.random.randn(dim))

    def judge(self, data, label):
        for x, i in zip(data, label):
            g_i = np.dot(self.a[i], x)
            for j in range(self.cls):
                if j != i and np.dot(self.a[j], x) > g_i:
                    return False
        return True

    def fit(self, data, label):
        data = preprocess_dataset(data, label, False)
        epoch, k = 0, 0
        num, dim = data.shape
        while True:
            i = label[k]
            gi = np.dot(self.a[i], data[k])
            for c in range(self.cls):
                if c != i and np.dot(self.a[c], data[k]) >= gi:
                    self.a[i] += data[k]
                    self.a[c] -= data[k]
            k = (k + 1) % num
            if k == 0:
                epoch += 1
                print("Epoch: %d" % epoch)
            if self.judge(data, label) or epoch == self.__max_epoch:
                break

    def predict(self, data, label):
        data = preprocess_dataset(data, label, False)
        num = label.shape[0]
        pre = np.empty_like(label)
        for idx, x in enumerate(data):
            max_v, max_i = -100000000, -1
            for i in range(self.cls):
                gi = np.dot(self.a[i], x)
                if gi > max_v:
                    max_v, max_i = gi, i
            pre[idx] = max_i

        correct = (pre == label).sum()
        print('[%d/%d] acc=%.2f%%' % (correct, num, correct / num * 100))


def run_tony_model():
    train_data, train_label = get_tony_dataset()
    perceptron = Perceptron(train_data.shape[1] + 1)
    perceptron.fit(train_data, train_label)
    print(perceptron.w)

    lsme = LMSE()
    lsme.fit(train_data, train_label)
    print(lsme.w)


def run_MNIST():
    (train_x, train_y), (test_x, test_y) = get_MNIST_dataset()
    keslersPerceptron = KeslersPerceptron(10, train_x.shape[1] + 1)
    keslersPerceptron.fit(train_x, train_y)
    keslersPerceptron.predict(test_x, test_y)


if __name__ == '__main__':
    run_tony_model()
    run_MNIST()
