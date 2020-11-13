import numpy as np


def get_tony_dataset(is_train1: bool):
    if is_train1:
        train_file, test_file = "data/Train1.csv", "data/Test1.csv"
    else:
        train_file, test_file = "data/Train2.csv", "data/Test2.csv"
    train_data = np.genfromtxt(train_file, delimiter=',')
    test_data = np.genfromtxt(test_file, delimiter=',')
    return train_data, test_data


def get_MNIST_dataset():
    train_file, train_labels_file = "data/TrainSamples.csv", "data/TrainLabels.csv"
    train_x = np.genfromtxt(train_file, delimiter=',')
    train_y = np.genfromtxt(train_labels_file, delimiter=',')
    sp_data = [[] for _ in range(10)]
    for i in range(len(train_y)):
        sp_data[int(train_y[i])].append(train_x[i])
    data, alpha = [], []
    for i in range(10):
        alpha.append(len(sp_data[i]) / len(train_y))
        data.append(np.array(sp_data[i]))

    test_file, test_labels_file = "data/TestSamples.csv", "data/TestLabels.csv"
    test_x = np.genfromtxt(test_file, delimiter=',')
    test_y = np.genfromtxt(test_labels_file, delimiter=',')
    return (data, alpha), (test_x, test_y)


def run_tony():
    gmm1, gmm2 = GMM(2), GMM(2)
    train1, test1 = get_tony_dataset(True)
    train2, test2 = get_tony_dataset(False)
    gmm1.fit(train1, len(train1))
    gmm2.fit(train2, len(train2))

    print('gmm1:')
    print('alpha: ', gmm1.alpha)
    print('avg: ', gmm1.avg)
    print('sigma: ', gmm1.sigma)

    print('gmm2:')
    print('alpha: ', gmm2.alpha)
    print('avg: ', gmm2.avg)
    print('sigma: ', gmm2.sigma)

    predict([gmm1, gmm2], test1, len(test1), np.full((len(test1),), 0), [0.5, 0.5])
    predict([gmm1, gmm2], test2, len(test2), np.full((len(test2),), 1), [0.5, 0.5])


def run_MNIST(kk):
    gmm_list = [GMM(kk, 1e-5) for _ in range(10)]
    (train_data, alpha), (test_x, test_y) = get_MNIST_dataset()
    for gmm, data in zip(gmm_list, train_data):
        gmm.fit(data, len(data))

    return predict(gmm_list, test_x, len(test_x), test_y, alpha)


def probability_density_function(x, avg, sigma):
    return np.exp(-0.5 * (np.dot(np.dot((x - avg).T, np.linalg.inv(sigma)), x - avg))) / \
           np.sqrt(np.linalg.det(sigma) * np.power(2 * np.pi, len(avg)))


class GMM:
    def __init__(self, k, err=1e-6):
        self.k = k
        self._err = err
        self.alpha, self.avg, self.sigma = None, None, None
        self.log_likelihood = 0

    def fit(self, data, num):
        copy_data = data.copy()
        np.random.shuffle(copy_data)
        sp_data = np.array_split(copy_data, self.k)
        self.alpha = np.repeat(1.0 / self.k, self.k)
        self.avg = np.array([np.mean(sp_data[i], axis=0) for i in range(self.k)])
        self.sigma = np.array([np.cov(sp_data[i].T) for i in range(self.k)])
        log_likelihood, epoch = 0, 0

        norm, res = np.empty((num, self.k), np.float), np.empty((num, self.k), np.float)
        while True:
            print("Epoch " + str(epoch))
            for i in range(num):
                x = data[i]
                for j in range(self.k):
                    norm[i][j] = probability_density_function(x, self.avg[j], self.sigma[j])
            log_likelihood = np.log(np.array(([np.dot(self.alpha, norm[i]) for i in range(num)]))).sum()
            print("log likelihood: " + str(log_likelihood))
            if abs(log_likelihood - self.log_likelihood) < self._err:
                self.log_likelihood = log_likelihood
                break

            # print("E step:")
            for i in range(num):
                normalize = np.dot(self.alpha.T, norm[i])
                for j in range(self.k):
                    res[i][j] = self.alpha[j] * norm[i][j] / normalize

            # print("M step:")
            for i in range(self.k):
                res_i = res.T[i]
                normalize = np.dot(res_i, np.ones(num))

                self.alpha[i] = normalize / num
                self.avg[i] = np.dot(res_i, data) / normalize
                diff = data - np.tile(self.avg[i], (num, 1))
                self.sigma[i] = np.dot((res_i.reshape(num, 1) * diff).T, diff) / normalize

            self.log_likelihood = log_likelihood
            epoch += 1


def predict(gmm_list, data, num, labels, alpha):
    log = np.empty((len(alpha), num), dtype=np.float)

    for idx, gmm in enumerate(gmm_list):
        norm = np.empty((num, gmm.k), np.float)
        for i in range(num):
            for j in range(gmm.k):
                norm[i][j] = probability_density_function(data[i], gmm.avg[j], gmm.sigma[j])
        log[idx] = np.array([np.dot(gmm.alpha, norm[i]) for i in range(num)]) * alpha[idx]

    pred = np.argmax(log, axis=0)
    correct_num = (pred == labels).sum()
    accuracy = correct_num / num
    print("{:d}, {:.2f}".format(correct_num, accuracy * 100))
    return correct_num, acc


if __name__ == '__main__':
    # run_tony()
    result = []
    for kkk in range(1, 6):
        cor, acc = run_MNIST(kkk)
        result.append((cor, acc))
    for re in result:
        print(str(re[0]) + " " + str(re[1]))
