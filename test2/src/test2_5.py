# -*- encoding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 读取discretization_data.xls，使用padans.cut()对数据进行0-3的离散处理
datafile = '../data/discretization_data.xls'
data = pd.read_excel(datafile)
data = data[u'肝气郁结证型系数'].copy()
d1 = pd.cut(data, k, labels = range(k))
print d1

# 处理后的数据保存在data_descret.xls中
# 实验：数据转化
data_descret_path = '../data/data_descret.xls'
w = [1.0*i/k for i in range(k+1)]
w = data.describe(percentiles = w)[4:4+k+1]
w[0] = w[0]*(1-1e-10)
d2 = pd.cut(data, w, labels = range(k))
d2.to_excel(data_descret_path, sheet_name='data_descret')
print d2

# 读取sales_data.xls，将字符数据转换成数值型数据，并将转换结果保存为data_tras.xls（data_transfer）
data_tras_path = '../data/data_tras.xls'
kmodel = KMeans(n_clusters = k, n_jobs = 4)
kmodel.fit(data.reshape((len(data), 1)))
c = pd.DataFrame(kmodel.cluster_centers_).sort(0)
w = pd.rolling_mean(c, 2).iloc[1:]
w = [0] + list(w[0]) + [data.max()]
d3 = pd.cut(data, w, labels = range(k))
d3.to_excel(data_tras_path, sheet_name='data_tras')
print d3
