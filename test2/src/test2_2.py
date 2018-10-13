# -*- encoding: utf-8 -*-

import os

# 读excel文件.
import xlrd

# 写excel文件.
import xlwt

# 使用lagrange插值算法..
from scipy.interpolate import lagrange


# 通过lagrange插值算法对数组进行插值.
def ploy_interpolate_column(s, n, k):
    """
    :param s: 需要进行插值的数组（一维）
    :param n: 需要计算插值的位置
    :param k: 取插值位置前后的参考数据个数
    :return: 需要插入的值
    """
    array = list(range(n - k, n)) + list(range(n + 1, n + 1 + k))
    y = []
    for index in array:
        y.append(s[index])
    return lagrange(array, y)(n)


# 定义文件路径
cater_sale_missing_path = '../data/catering_sale_missing.xls'
missing_data_processing_path = '../data/missing_data_processing.xls'

# 读取cater_sale_missing.xls文件内容
cater_sale_missing_data = xlrd.open_workbook(cater_sale_missing_path)
cater_sale_missing_sheet = cater_sale_missing_data.sheets()[0]
cater_sale_rows_count = cater_sale_missing_sheet.nrows

# 读取cater_sale_missing.xls文件的列名
first_column_name = cater_sale_missing_sheet.row_values(0)[0]
second_column_name = cater_sale_missing_sheet.row_values(0)[1]

# 读取cater_sale_missing中的数据
first_column_data = []
second_column_data = []
for i in range(1, cater_sale_rows_count):
    first_column_data.append(cater_sale_missing_sheet.row_values(i)[0])
    second_column_data.append(cater_sale_missing_sheet.row_values(i)[1])

# 获取缺失数据的索引
missing_data_indexes = []
for i in range(second_column_data.__len__()):
    if second_column_data[i] == '':
        missing_data_indexes.append(i)

# 对缺失的数据通过lagrange插值算法进行插值
k = 5
for i in missing_data_indexes:
    second_column_data[i] = round(ploy_interpolate_column(second_column_data, i, k), 1)

# 写入missing_data_processing.xls文件中
if os.path.exists(missing_data_processing_path):
    os.remove(missing_data_processing_path)

missing_data_processing_data = xlwt.Workbook(encoding='utf-8')
missing_data_processing_sheet = missing_data_processing_data.add_sheet('missing_data_processing')

missing_data_processing_style = xlwt.XFStyle()
missing_data_processing_style.num_format_str = 'YYY/M/D'

missing_data_processing_sheet.write(0, 0, label=first_column_name)
missing_data_processing_sheet.write(0, 1, label=second_column_name)

for i in range(first_column_data.__len__()):
    missing_data_processing_sheet.write(i + 1, 0, first_column_data[i], missing_data_processing_style)
    missing_data_processing_sheet.write(i + 1, 1, second_column_data[i])

missing_data_processing_data.save(missing_data_processing_path)

print('Completed!')
