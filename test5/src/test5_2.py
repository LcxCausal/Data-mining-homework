"""
数据集:
    dataSet..txt共1000个样本，包括以下三个特征：每年飞行的里程数、玩视频游戏所耗时间百分数，
    以及每周消费的冰淇淋公升数。希望根据上面三个特征，将上述人群分成1、2、3类。
实验要求：
    1）	读取数据
    2）	归一化特征值
    3）	实现KNN算法
    4）	根据数据集提供的类别，计算误差比例，并统计各类别数量
"""

# -*- encoding='utf-8' -*-

from numpy import *
import operator
from collections import Counter
import matplotlib
import matplotlib.pyplot as plt


# 导入特征数据
def file2matrix(filename):
    fr = open(filename)
    contain = fr.readlines()  # 读取文件的所有内容
    count = len(contain)
    return_mat = zeros((count, 3))
    class_label_vector = []
    index = 0
    for line in contain:
        line = line.strip()  # 截取所有的回车字符
        list_from_line = line.split('\t')
        return_mat[index, :] = list_from_line[0:3]  # 选取前三个元素，存储在特征矩阵中
        class_label_vector.append(list_from_line[-1])  # 将列表的最后一列存储到向量classLabelVector中
        index += 1

    # 将列表的最后一列由字符串转化为数字，便于以后的计算
    dict_class_label = Counter(class_label_vector)
    class_label = []
    kind = list(dict_class_label)
    for item in class_label_vector:
        if item == kind[0]:
            item = 1
        elif item == kind[1]:
            item = 2
        else:
            item = 3
        class_label.append(item)
    return return_mat, class_label  # 将文本中的数据导入到列表


# 归一化数据,保证特征等权重
def autoNorm(data_set):
    min_values = data_set.min(0)
    max_values = data_set.max(0)
    ranges = max_values - min_values
    norm_data_set = zeros(shape(data_set))  # 建立与dataSet结构一样的矩阵
    m = data_set.shape[0]
    for i in range(1, m):
        norm_data_set[i, :] = (data_set[i, :] - min_values) / ranges
    return norm_data_set, ranges, min_values


# KNN算法
def classify(input, data_set, label, k):
    data_size = data_set.shape[0]
    # 计算欧式距离
    diff = tile(input, (data_size, 1)) - data_set
    sqdiff = diff ** 2
    square_dist = sum(sqdiff, axis=1)  # 行向量分别相加，从而得到新的一个行向量
    dist = square_dist ** 0.5

    # 对距离进行排序
    sorted_dist_index = argsort(dist)  # argsort()根据元素的值从大到小对元素进行排序，返回下标

    class_count = {}
    for i in range(k):
        vote_label = label[sorted_dist_index[i]]
        # 对选取的K个样本所属的类别个数进行统计
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    # 选取出现的类别次数最多的类别
    max_count = 0
    for key, value in class_count.items():
        if value > max_count:
            max_count = value
            classes = key
    return classes


# 测试(选取10%测试）
def datingTest(file):
    rate = 0.10
    dating_data_mat, dating_labels = file2matrix(file)
    norm_mat, ranges, min_vals = autoNorm(dating_data_mat)
    m = norm_mat.shape[0]
    test_num = int(m * rate)
    error_count = 0.0
    for i in range(1, test_num):
        classify_result = classify(norm_mat[i, :], norm_mat[test_num:m, :], dating_labels[test_num:m], 3)
        print("分类后的结果为:,", classify_result)
        print("原结果为：", dating_labels[i])
        if classify_result != dating_labels[i]:
            error_count += 1.0
    print("误分率为:", (error_count / float(test_num)))


# 预测函数
def classifyPerson(file):
    result_list = ['一点也不喜欢', '有一丢丢喜欢', '灰常喜欢']
    percent_tats = float(input("玩视频所占的时间比?"))
    miles = float(input("每年获得的飞行常客里程数?"))
    ice_cream = float(input("每周所消费的冰淇淋公升数?"))
    dating_data_mat, dating_labels = file2matrix(file)
    norm_mat, ranges, min_vals = autoNorm(dating_data_mat)
    in_arr = array([miles, percent_tats, ice_cream])
    classifier_result = classify((in_arr - min_vals) / ranges, norm_mat, dating_labels, 3)
    print("你对这个人的喜欢程度:", result_list[classifier_result - 1])


file = '../data/dataSet.txt'

# 绘图（可以直观的表示出各特征对分类结果的影响程度）
datingDataMat, datingLabels = file2matrix(file)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
plt.show()

classifyPerson(file)
