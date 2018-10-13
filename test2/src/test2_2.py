# -*- encoding: utf-8 -*-

import os

# 读excel文件.
import xlrd

# 写excel文件.
import xlwt

# 使用lagrange插值算法..
from scipy.interpolate import lagrange


# 通过lagrange插值算法对数组进行插值.
def ploy_interpolate_column(data_array, index, range_count):
    """
    :param data_array: 需要进行插值的数组（一维）
    :param index: 需要计算插值的位置
    :param range_count: 取插值位置前后的参考数据个数
    :return: 需要插入的值
    """
    array = list(range(index - range_count, index)) + list(range(index + 1, index + 1 + range_count))
    y = []
    for index in array:
        y.append(data_array[index])
    return lagrange(array, y)(index)


# 定义文件路径
cater_sale_all_path = '../data/catering_sale_all.xls'
missing_data_processing_path = '../data/missing_data_processing.xls'

# 读取cater_sale_missing.xls文件内容
cater_sale_all_data = xlrd.open_workbook(cater_sale_all_path)
cater_sale_all_sheet = cater_sale_all_data.sheets()[0]
cater_sale_rows_count = cater_sale_all_sheet.nrows
cater_sale_columns_count = cater_sale_all_sheet.ncols

# 读取cater_sale_missing.xls文件的列名
cater_sale_column_names = []
cater_sale_first_row_data = cater_sale_all_sheet.row_values(0)
for i in range(cater_sale_columns_count):
    cater_sale_column_names.append(cater_sale_first_row_data[i])

# 读取cater_sale_missing中的数据
cater_sale_column_data = []
for i in range(cater_sale_columns_count):
    column_data = []
    for j in range(1, cater_sale_rows_count):
        column_data.append(cater_sale_all_sheet.row_values(j)[i])
    cater_sale_column_data.append(column_data)

# 获取缺失数据的索引
missing_data_indexes = []
for i in range(cater_sale_column_data.__len__()):
    missing_column_data_indexes = []
    for j in range(cater_sale_rows_count - 1):
        if str(cater_sale_column_data[i][j]).strip() == '':
            missing_column_data_indexes.append(j)
    missing_data_indexes.append(missing_column_data_indexes)

# 对缺失的数据通过lagrange插值算法进行插值
k = 2
for i in range(missing_data_indexes.__len__()):
    missing_column_data_indexes = missing_data_indexes[i]
    for j in missing_column_data_indexes:
        cater_sale_column_data[i][j] = round(ploy_interpolate_column(cater_sale_column_data[i], j, k), 1)

# 写入missing_data_processing.xls文件中
if os.path.exists(missing_data_processing_path):
    os.remove(missing_data_processing_path)

missing_data_processing_data = xlwt.Workbook(encoding='utf-8')
missing_data_processing_sheet = missing_data_processing_data.add_sheet('missing_data_processing')

missing_data_processing_style = xlwt.XFStyle()
missing_data_processing_style.num_format_str = 'YYY/M/D'

for i in range(cater_sale_column_names.__len__()):
    missing_data_processing_sheet.write(0, i, label=cater_sale_column_names[i])

for i in range(cater_sale_column_data.__len__()):
    column_data = cater_sale_column_data[i]
    for j in range(column_data.__len__()):
        if i == 0:
            missing_data_processing_sheet.write(j + 1, i, column_data[j], missing_data_processing_style)
        else:
            missing_data_processing_sheet.write(j + 1, i, column_data[j])

missing_data_processing_data.save(missing_data_processing_path)



print('Completed!')
