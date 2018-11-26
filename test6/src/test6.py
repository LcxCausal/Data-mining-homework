# coding=utf-8

"""
1）	对所有窃漏电用户及正常用户的电量、告警数据和该用户再当天是否窃漏电的标识，按窃漏电评价指标处理并选取其中291个样本数据，得到专家样本，数据见model.xls。分别使用神经网络和决策树实现对用电用户进行分类预测。
2）	把model.xls按80%训练数据，20%样本数据进行分割。
3）	使用keras库以及训练数据构建神经网络模型，使用predict函数和构建的神经网络模型对样本数据进行分类。
4）	使用sklearn库以及训练数据构建决策树模型，并对样本数据进行分类。
5）	对模型分类效果进行对比和分析。
6）	提交源代码
"""

from random import shuffle
from keras.models import Sequential
# 导入神经网络层函数、激活函数
from keras.layers.core import Dense, Activation
import pandas as pd
import numpy as np

# 读取 model.xls 文件中的数据

data_path = '../data/model.xls'
data = pd.read_excel(data_path, float_format='%.1f')
column_names = np.array(data.columns)

column_data = []
for column_name in column_names:
    column_data.append(data[column_name])


# 数据归一化处理

def get_min_and_max_data_from_two_dimensional_array(input_data):
    columns_min_data = []
    columns_max_data = []
    for col_data in input_data:
        min_data = col_data[0]
        max_data = col_data[0]
        for row_data in col_data:
            if min_data > row_data:
                min_data = row_data
            if max_data < row_data:
                max_data = row_data
        columns_min_data.append(min_data)
        columns_max_data.append(max_data)
    return columns_min_data, columns_max_data


def normalize_data(input_data):
    output_data = []
    columns_min_data, columns_max_data = get_min_and_max_data_from_two_dimensional_array(input_data)
    for col in range(input_data.__len__()):
        min_data = columns_min_data[col]
        max_data = columns_max_data[col]
        max_diff = max_data - min_data
        tmp_output_data = []
        for row in range(input_data[col].__len__()):
            tmp_output_data.append((input_data[col][row] - min_data) / max_diff)
        output_data.append(tmp_output_data)
    return output_data


column_data = normalize_data(column_data)


# 将数据的 80% 分为训练数据， 20% 分为测试数据

def sort_data_to_training_and_testing(input_data):
    tmp_out_training_data = []
    tmp_out_testing_data = []
    for col_data in input_data:
        col_len = col_data.__len__()
        training_data_count = col_len * 0.8
        tmp_training_data = []
        tmp_testing_data = []
        for row in range(col_len):
            if 0 <= row <= training_data_count:
                tmp_training_data.append(col_data[row])
            elif training_data_count < row <= col_len:
                tmp_testing_data.append(col_data[row])
        tmp_out_training_data.append(tmp_training_data)
        tmp_out_testing_data.append(tmp_testing_data)
    out_training_data = []
    out_testing_data = []
    for row in range(tmp_out_training_data[0].__len__()):
        tmp_data = []
        for col in range(tmp_out_training_data.__len__()):
            tmp_data.append(tmp_out_training_data[col][row])
        out_training_data.append(tmp_data)
    for row in range(tmp_out_testing_data[0].__len__()):
        tmp_data = []
        for col in range(tmp_out_testing_data.__len__()):
            tmp_data.append(tmp_out_testing_data[col][row])
        out_testing_data.append(tmp_data)
    return out_training_data, out_testing_data


training_data, testing_data = sort_data_to_training_and_testing(column_data)

# 构建 BP 神经网络

training_data = pd.DataFrame(training_data)
testing_data = pd.DataFrame(testing_data)
training_data = training_data.as_matrix()
testing_data = testing_data.as_matrix()
shuffle(training_data)
shuffle(testing_data)

model_path = "../data/model.model"
model = Sequential()
model.add(Dense(input_dim=3, units=10))
model.add(Activation('relu'))
model.add(Dense(input_dim=10, units=1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(training_data[:, :3], training_data[:, 3], epochs=10, batch_size=1)
model.save_weights(model_path)

from sklearn.metrics import confusion_matrix  # 导入混淆矩阵函数

predict_result = model.predict_classes(training_data[:, :3]).reshape(len(training_data))  # 预测结果变形
cm = confusion_matrix(training_data[:, 3], predict_result)  # 混淆矩阵

import matplotlib.pyplot as plt  # 导入作图库

plt.matshow(cm, cmap=plt.cm.Greens)  # 画混淆矩阵图，配色风格使用cm.Greens
plt.colorbar()  # 颜色标签

for x in range(len(cm)):  # 数据标签
    for y in range(len(cm)):
        plt.annotate(cm[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')

plt.ylabel('Truelabel')  # 坐标轴标签
plt.xlabel('Predicted label')  # 坐标轴标签
plt.show()  # 显示作图结果

from sklearn.metrics import roc_curve  # 导入ROC曲线函数

predict_result = model.predict(testing_data[:, :3]).reshape(len(testing_data))
fpr, tpr, thresholds = roc_curve(testing_data[:, 3], predict_result, pos_label=1)
plt.plot(fpr, tpr, linewidth=2, label='ROC of LM')  # 作出ROC曲线
plt.xlabel('False Positive Rate')  # 坐标轴标签
plt.ylabel('True Positive Rate')  # 坐标轴标签
plt.ylim(0, 1.05)  # 边界范围
plt.xlim(0, 1.05)  # 边界范围
plt.legend(loc=4)  # 图例
plt.show()  # 显示作图结果
