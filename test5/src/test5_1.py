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
import os
import numpy as np
import pandas as pd


# 1、	读入Iris.csv数据，并将数据的70%做为训练数据，30%做为测试数据
iris_file = '../data/iris.csv'
iris_data = pd.read_csv(iris_file)
iris_data_shape = iris_data.shape
iris_row_count = iris_data_shape[0]

1

