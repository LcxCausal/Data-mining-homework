# -*- encoding: utf-8 -*-

import pandas as pd
import numpy as np

normalistion_data_path = '../data/normalization_data.xls'

# 读取normalization_data.xls
data = pd.read_excel(normalistion_data_path, header=None)

# 最小-最大规范化数据
a = (data - data.min())/(data.max() - data.min())

# Z-score规范化(零-均值规范化)
b = (data - data.mean())/data.std()

# 小数定标规范化
c = data/10**np.ceil(np.log10(data.abs().max()))

# 输出所有规范化结果
print(a)
print(b)
print(c)
