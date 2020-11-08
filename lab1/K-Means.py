import numpy as np
import random
from tqdm import tqdm


def run_tony():
    model = KMeans(2)
    tony_dataset = np.array([(0, 0), (1, 0), (0, 1), (1, 1),
                             (2, 1), (1, 2), (2, 2), (3, 2),
                             (6, 6), (7, 6), (8, 6), (7, 7),
                             (8, 7), (9, 7), (7, 8), (8, 8),
                             (9, 8), (8, 9), (9, 9)])
    model.fit(tony_dataset)
    for i, category in enumerate(model.categories):
        print(i)
        print(category)


def run_MNIST(k: int, file_nums: str):
    model = KMeans(k)
    x, y = get_MNIST_dataset()
    model.fit(x)
    result_file = 'result/result' + file_nums + '.csv'
    with open(result_file, 'w', encoding='utf8') as f:
        for center in model.init_centers:
            f.write(str(center) + '\n')
        f.write(',0,1,2,3,4,5,6,7,8,9\n')
        for i, idx_category in enumerate(model.index_categories):
            real_y = [0] * k
            f.write('Cluster' + str(i) + ',' + str(len(idx_category)))
            for idx in idx_category:
                real_y[y[idx]] += 1
            for p in range(k):
                f.write(',' + str(real_y[p]))
            f.write('\n')


def get_MNIST_dataset():
    samples_file, labels_file = "data/ClusterSamples.csv", "data/SampleLabels.csv"
    x = np.genfromtxt(samples_file, delimiter=',')
    y = np.genfromtxt(labels_file, delimiter=',', dtype=np.int8)
    return x, y


class KMeans(object):

    def __init__(self, k, max_epoch=300):
        # categories是聚类后的数据，index_categories是聚类后数据的index
        self.categories, self.index_categories = [], []
        # init_centers是随机初始化的中心，centers是迭代过程中的中心
        self.init_centers, self.centers = [], []
        # k个类别
        self._k = k
        # 最大训练轮次
        self._max_epoch = max_epoch

    def fit(self, data):
        # 随机初始化中心
        random_center_idx = random.sample(range(len(data)), self._k)
        self.init_centers = [data[i] for i in random_center_idx]
        self.centers = self.init_centers
        last_idx_cate = [[] for _ in range(self._k)]
        # 训练
        for epoch in range(self._max_epoch):
            print('Epoch: ' + str(epoch))
            self.categories, self.index_categories = [[] for _ in range(self._k)], [[] for _ in range(self._k)]
            # 对每个点进行聚类，选取离其最近的中心为其分类结果
            for idx, d in enumerate(tqdm(data, ncols=100)):
                distances = [np.linalg.norm(d - self.centers[i]) for i in range(self._k)]
                category = distances.index(min(distances))
                self.categories[category].append(d)
                self.index_categories[category].append(idx)
            # 重新计算中心
            self.centers = [np.average(self.categories[i], axis=0) for i in range(self._k)]
            # 判断聚类结果是否不再变化，不变则退出训练
            zoom_out = True
            for i in range(self._k):
                if set(last_idx_cate[i]) != set(self.index_categories[i]):
                    zoom_out = False
            last_idx_cate = self.index_categories
            if zoom_out:
                break


if __name__ == '__main__':
    run_tony()
    for num in range(10):
        run_MNIST(10, str(num))
