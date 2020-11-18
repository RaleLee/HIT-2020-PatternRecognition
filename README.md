# HIT-2020-PatternRecognition

哈工大2020秋季学期模式识别研究生课实验

反正是选修课，求求不要卷，开心及格不好么

如果有问题，欢迎PR，issues交流~~

## Lab1 K-Means

需要的库：numpy tqdm
- 这里可能会遇到聚类结果不好的问题，但因为初始聚类中心是随机初始化的，其实很正常。如果想要提高聚类效果，可以先使用层次聚类的方法初始化较好的中心。

## Lab2 GMM

需要的库：numpy
- tony model很正常
- MNIST一句话解释：用10个类别的数据，去分别训练识别10个类别的GMM，每个GMM的高斯数按照实验手册要求从1变化到5.

## Lab3 线性分类器

需要的库：numpy
- 感知器和LMSE很正常，伪代码同ppt
- 多类别分类：使用Kesler‘s Perceptron方法，伪代码同ppt

## Exam Lab 神经网络分类器

需要的库：Pytorch numpy sklearn tqdm
- 网络结构 encoder -> hidden_layer -> decoder
- 训练执行train.py GPU和CPU都可以
- 考试执行test.py 需要预先训练好的model文件 model文件可以是GPU上训练好的也可以是CPU上训练好的
- 没有什么花里胡哨的，选修课，不需要卷，所以没有调参，但是所有参数都是可以调的