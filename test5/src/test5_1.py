"""
数据集：
    Iris也称鸢尾花卉数据集，是一类多重变量分析的数据集。该数据集是4个最流行的机器学习数据集之一。
    通过花萼长度，花萼宽度，花瓣长度，花瓣宽度4个属性预测鸢尾花卉属于（Setosa，Versicolour，Virginica）三个种类中的哪一类。
实验要求：
    1、	读入Iris.csv数据，并将数据的70%做为训练数据，30%做为测试数据；
    2、	根据KNN算法思想，实现KNN算法，并使用训练数据训练KNN分类模型；
    3、	使用剩余30%数据对KNN分类模型进行测试，并计算出分类精度。（可考虑使用可视化界面进行分类效果分析）
    4、	可以直接用sklearn库中的neighbors.KNeighborsClassifier()
提示：
    1、	from random import shuffle  --shuffle()
    2、	可视化模块
"""

# -*- encoding='utf-8' -*-
import csv

# 1、读入Iris.csv数据，并将数据的70%做为训练数据，30%做为测试数据
iris_file = '../data/iris.csv'
iris_data = []
training_data = []
testing_data = []

with open(iris_file) as iris_file_data:
    data = csv.reader(iris_file_data)
    for d in data:
        iris_data.append(d)

iris_data_rows = iris_data.__len__() - 1
training_data_rows = int(iris_data_rows * 0.7)
testing_data_rows = iris_data_rows - training_data_rows

for i in range(1, training_data_rows + 1):
    training_data.append(iris_data[i])

for i in range(training_data_rows + 1, iris_data_rows + 1):
    testing_data.append(iris_data[i])

# 2、根据KNN算法思想，实现KNN算法，并使用训练数据训练KNN分类模型
from numpy import *
import KNN


def classify(input_data, data_set, labels):
    data_size = data_set.shape[0]
    diff = tile(input_data, (data_size, 1)) - data_set
    sq_diff = diff ** 2
    square_dist = sum(sq_diff, axis=1)
    dist = square_dist ** 0.5
    sorted_dist_index = argsort(dist)
    class_count = {}
    k = data_set.__len__();

    for i in range(k):
        vote_label = labels[sorted_dist_index[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    max_count = 0
    for key, value in class_count.items():
        if value > max_count:
            max_count = value
            classes = key

    return classes


column_names = iris_data[0]
data_set, labels = createDataSet()
knn_data = classify(training_data, data_set, labels)

1
