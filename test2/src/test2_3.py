# -*- encoding: utf-8 -*-

# 使用Pandas的特征统计函数
import pandas

missing_data_processing_path = '../data/missing_data_processing.xls'

# 计算任意两款菜式之间的相关系数
data = pandas.read_excel(missing_data_processing_path, index_col=u'日期')
print(data.corr())

# 只计算“百合酱蒸凤爪”与其他菜式的相关系数
print(data.corr()[u'百合酱蒸凤爪'])

# 计算“百合酱蒸凤爪”与“翡翠蒸香茜饺”的相关系数
print(data[u'百合酱蒸凤爪'].corr(data[u'翡翠蒸香茜饺']))

print('Completed!')
