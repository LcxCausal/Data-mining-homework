# air_data.csv为某航空公司系统内的客户信息，根据末次飞行日期（LAST_FLIGHT_DATE）, 抽取包含2012-04-01至2014-03-31内所有
# 乘客的详细数据，并存入air_data_part.xls

import pandas as pd
import numpy as np


# 获取 LAST_FLIGHT_DATE 的索引数
def get_column_index(column_name, data_df):
    air_data_columns = data_df.columns

    for col in range(0, air_data_columns.__len__()):
        if air_data_columns[col].__eq__(column_name):
            return col

    return -1


# 将 string 类型的 date 转换成 int 类型的元组
def get_special_date(date):
    s_date = date.split(' ')[0].split('/')
    return int(s_date[0]), int(s_date[1]), int(s_date[2])


# 抽取包含2012-04-01至2014-03-31内所有乘客的详细数据
def get_air_data_part(data_df, column_index, begin, end):
    begin_year, begin_month, begin_day = get_special_date(begin)
    end_year, end_month, end_day = get_special_date(end)

    air_data_shape = data_df.shape
    air_data_values = data_df.get_values()

    data_part = []

    for row in range(0, air_data_shape[0]):
        row_air_data = air_data_values[row]
        column_value = row_air_data[column_index]
        current_year, current_month, current_day = get_special_date(column_value)

        if begin_year <= current_year <= end_year:
            if begin_year.__eq__(current_year) and current_month >= begin_month and current_day >= begin_day \
                    or end_year.__eq__(current_year) and current_month <= end_month and current_day <= end_day \
                    or begin_year < current_day < end_year:
                data_part.append(row_air_data)

    return data_part


air_data_path = '../data/air_data.csv'
air_data_df = pd.read_csv(air_data_path)
air_data = air_data_df.get_values()
last_flight_data_column_index = get_column_index('LAST_FLIGHT_DATE', air_data_df)

air_data_part = get_air_data_part(air_data_df, last_flight_data_column_index, '2012/04/01', '2014/03/31')
air_data_part_path = '../data/air_data_part.xls'
air_data_part_df = pd.DataFrame(air_data_part, columns=air_data_df.columns)
air_data_part_df.to_excel(air_data_part_path)

# 数据分析：统计air_data_part.xls中每列属性观测值中空值的个数、最大值、最小值的个数，并存入data_analysis.xls
part_data_df = pd.read_excel(air_data_part_path)
part_data_columns = part_data_df.columns
part_data_shape = part_data_df.shape
part_data_values = part_data_df.get_values()

columns_null_count = []
columns_max_count = []
columns_min_count = []
columns_max = []
columns_min = []

for col_index in range(0, part_data_shape[1]):
    columns_null_count.append(0)
    columns_max_count.append(0)
    columns_min_count.append(0)

    try:
        columns_min.append(float(part_data_values[0][col_index]))
        columns_max.append(float(part_data_values[0][col_index]))
    except:
        columns_min.append(0)
        columns_max.append(0)

    for row_index in range(1, part_data_shape[0]):
        try:
            if columns_min[col_index] > float(part_data_values[row_index][col_index]):
                columns_min[col_index] = float(part_data_values[row_index][col_index])
            if columns_max[col_index] < float(part_data_values[row_index][col_index]):
                columns_max[col_index] = float(part_data_values[row_index][col_index])
        except:
            columns_min[col_index] = 0
            columns_max[col_index] = 0

for row_index in range(0, part_data_shape[0]):
    row_data = part_data_values[row_index]
    for col_index in range(0, part_data_shape[1]):
        col_data = row_data[col_index]

        if col_data == '' or col_data is None:
            columns_null_count[col_index] += 1
        else:
            try:
                float_col_data = float(col_data)
                if float_col_data.__eq__(float(columns_min[col_index])):
                    columns_min_count[col_index] += 1
                if float_col_data.__eq__(float(columns_max[col_index])):
                    columns_max_count[col_index] += 1
            except:
                columns_min_count[col_index] = 0
                columns_max_count[col_index] = 0

data_analysis_path = '../data/data_analysis.xls'
data_analysis_df = pd.DataFrame(list([np.array(columns_null_count), np.array(columns_max_count),
                                      np.array(columns_min_count)]), columns=part_data_columns)
data_analysis_df.to_excel(data_analysis_path)

# 数据清理：给定始数据air_data.csv, 去除原始数据中缺失值记录。处理方法如下：保留票价非零，或者平均折扣率与飞行公里数同时为0的记录
# index1=data[ ‘SUM_YR’] !=0
# index2=data[‘SEG_KM_SUM’] ==0 & data[‘AVG_DISCOUNT’]==0
# data = data[ index1 | index2]
# 将清理后的数据保存为data_cleaned.csv 或data_cleaned.xls
air_data_df = pd.read_csv(air_data_path)
data_cleaned = []
air_data_columns = air_data_df.columns
air_data_values = air_data_df.get_values()
air_data_shape = air_data_df.shape

col_sum_yr_index = get_column_index('SUM_YR', air_data_df)
col_seg_km_sum_index = get_column_index('SEG_KM_SUM', air_data_df)
col_avg_discount_index = get_column_index('AVG_DISCOUNT', air_data_df)

for row_index in range(0, air_data_shape[0]):
    row_air_data = air_data_values[row_index]
    if float(row_air_data[col_sum_yr_index]) != 0 or \
            float(row_air_data[col_seg_km_sum_index]) == 0 and float(row_air_data[col_avg_discount_index]) == 0:
        data_cleaned.append(row_air_data)

data_cleaned_path = '../data/data_cleaned.xls'
data_cleaned_df = pd.DataFrame(data_cleaned, columns=air_data_columns)
data_cleaned_df.to_excel(data_cleaned_path)


# 数据抽取：根据原始数据air_data.csv,从24个属性抽取下面六个属性：LOAD_TIME, FFP_DATE, LAST_TO_END, FLIGHT_COUNT, SEG_KM_SUM, AVG_DISCOUNT
# 【要求2】：将属性选择后的数据保存为 zscoredata.xls
zs_core_data_path = '../data/zscoredata.xls'
zs_core_data = []
zs_core_data_columns = np.array(['LOAD_TIME', 'FFP_DATE', 'LAST_TO_END', 'FLIGHT_COUNT', 'SEG_KM_SUM', 'AVG_DISCOUNT'])
zs_core_data_column_indexes = []

for zs_core_data_column in zs_core_data_columns:
    zs_core_data_column_indexes.append(get_column_index(zs_core_data_column, air_data_df))

for row_index in range(0, air_data_shape[0]):
    row_air_data = air_data_values[row_index]
    row_zs_core_data = []

    for col_index in range(0, air_data_shape[1]):
        if zs_core_data_column_indexes.__contains__(col_index):
            row_zs_core_data.append(row_air_data[col_index])

        zs_core_data.append(row_zs_core_data)

zs_core_data_df = pd.DataFrame(zs_core_data, columns=zs_core_data_columns)
zs_core_data_df.to_excel(zs_core_data_path)

