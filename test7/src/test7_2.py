# air_data.csv为某航空公司系统内的客户信息，根据末次飞行日期（LAST_FLIGHT_DATE）, 抽取包含2012-04-01至2014-03-31内所有
# 乘客的详细数据，并存入air_data_part.xls

from random import randint
import pandas as pd
import numpy as np


# 获取 LAST_FLIGHT_DATE 的索引数
def get_column_index(column_name, data_df):
    data_columns = data_df.columns

    for col in range(0, data_columns.__len__()):
        if data_columns[col].__eq__(column_name):
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

    data_shape = data_df.shape
    data_values = data_df.get_values()

    data_part = []

    for row in range(0, data_shape[0]):
        r_data = data_values[row]
        column_value = r_data[column_index]
        current_year, current_month, current_day = get_special_date(column_value)

        if begin_year <= current_year <= end_year:
            if begin_year.__eq__(current_year) and current_month >= begin_month and current_day >= begin_day \
                    or end_year.__eq__(current_year) and current_month <= end_month and current_day <= end_day \
                    or begin_year < current_day < end_year:
                data_part.append(r_data)

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
zs_core_data_columns = np.array(['LOAD_TIME', 'FFP_DATE', 'LAST_TO_END', 'FLIGHT_COUNT', 'SEG_KM_SUM', 'avg_discount'])
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

# 数据变换：采用z-core对上面数据进行标准化处理, 具体计算如下：
# （1）L = LOAD_TIME-FFP_DATE
# 会员入会时间距观测窗口结束的月数=观测窗口的结束时间-入会时间（单位：月）
#  (2) R = LAST_TO_END
# 客户最近一次乘坐公司飞机距观测窗口结束的月数=最后一次乘机时间至观察窗口末端时长（单位：月）
#  (3) F = FLIGHT_COUNT
# 客户在观测窗口内乘坐公司飞机的次数=观测窗口的飞行次数（单位：次）
#  (4) M = SEG_KM_SUM
# 客户在观测时间内在公司累计的飞行里程=观测窗口的总飞行公里数（单位：公里）
#  (5) C = AVG_DISCOUNT
# 客户在观测时间内乘坐舱位所对应的折扣系数的平均值=平均折扣率（单位：无）
# 【要求3】：将变换后的数据保存为 zscoreddata.xls
zs_cored_data_columns = np.array(['L', 'R', 'F', 'M', 'C'])
l = []
r = []
f = []
m = []
c = []
zs_cored_data = []

load_time_index = 0
ffp_date_index = 1
last_to_end_index = 2
flight_count_index = 3
seg_km_sum_index = 4
avg_discount_index = 5

for row_zs_core_data in zs_core_data:
    r_load_time = row_zs_core_data[load_time_index]
    r_ffp_date = row_zs_core_data[ffp_date_index]
    r_last_to_end = row_zs_core_data[last_to_end_index]
    r_flight_count = row_zs_core_data[flight_count_index]
    r_seg_km_sum = row_zs_core_data[seg_km_sum_index]
    r_avg_discount = row_zs_core_data[avg_discount_index]

    r_load_time_year, r_load_time_month, r_load_time_day = get_special_date(r_load_time)
    r_ffp_date_year, r_ffp_date_month, r_ffp_date_day = get_special_date(r_ffp_date)
    l_month_count = (r_load_time_year - r_ffp_date_year) * 12 + r_load_time_month - r_ffp_date_month

    if r_load_time_day - r_ffp_date_day < 0:
        l_month_count -= 1

    l.append(l_month_count.__str__())
    r.append(r_last_to_end)
    f.append(r_flight_count)
    m.append(r_seg_km_sum)
    c.append(r_avg_discount)

for row in range(0, l.__len__()):
    zs_cored_data.append(np.array([l[row], r[row], f[row], m[row], c[row]]))

zs_cored_data_path = "../data/zscoreddata.xls"
zs_cored_data_df = pd.DataFrame(zs_cored_data,
                                columns=zs_cored_data_columns)
zs_cored_data_df.to_excel(zs_cored_data_path)


# 构建模型：采用k-mean聚类算法进行模型构建
# 【要求4】：将聚类类别、聚类个数、聚类中心结果保存为output1.xls，格式如下：
# 【要求5】：输出原始数据(变换后数据)及其类别，并将结果保存为k-meanfile.xls
# 【要求6】：根据output1.xls结果，对客户群进行特征分析

# 数据归一化处理

def get_min_and_max_data_from_two_dimensional_array(input_data):
    columns_min_data = []
    columns_max_data = []
    for col_d in input_data:
        min_data = col_d[0]
        max_data = col_d[0]
        for row_d in col_d:
            if min_data > row_d:
                min_data = row_d
            if max_data < row_d:
                max_data = row_d
        columns_min_data.append(min_data)
        columns_max_data.append(max_data)
    return columns_min_data, columns_max_data


def normalize_data(input_data):
    output_data = []
    columns_min_data, columns_max_data = get_min_and_max_data_from_two_dimensional_array(input_data)
    for col_i in range(input_data.__len__()):
        min_data = columns_min_data[col_i]
        max_data = columns_max_data[col_i]
        max_diff = max_data - min_data
        tmp_output_data = []
        for row_i in range(input_data[col_i].__len__()):
            tmp_output_data.append((input_data[col_i][row_i] - min_data) / max_diff)
        output_data.append(tmp_output_data)
    return output_data


# 实现 k_means 算法
def k_means_core(input_data, input_data_parameters, centers, cols):
    # 对中心点外的数据进行分类
    output_data = []
    for o_i in range(0, centers.__len__()):
        output_data.append([])
    for i_data_para_index in range(0, input_data_parameters.__len__()):
        min_index = 0
        min_distance = max(centers)
        for c_i in range(0, centers.__len__()):
            tmp_min_distance = abs(input_data_parameters[i_data_para_index] - centers[c_i])
            if min_distance > tmp_min_distance:
                min_distance = tmp_min_distance
                min_index = c_i
        output_data[min_index].append(input_data.values[i_data_para_index])
    # 计算新簇的中心值
    tmp_centers = []
    for op_data in output_data:
        sum_value = 0
        for o_data in op_data:
            for o_col in o_data[0:cols]:
                sum_value += o_col
        tmp_centers.append(sum_value / op_data.__len__())
    # 判断中心点是否稳定，若不稳定则根据新的中心点重新进行分类
    is_regroup = False
    for i in range(0, centers.__len__()):
        if not tmp_centers[i].__eq__(centers[i]):
            is_regroup = True
            break
    if is_regroup:
        output_data = k_means_core(input_data, input_data_parameters, tmp_centers, cols)

    return output_data


def k_means(input_data, k, cols):
    # 生成 k 个随机数
    indexes = []
    while not indexes.__len__().__eq__(k):
        r_int = randint(0, input_data.shape[0])
        if not indexes.__contains__(r_int):
            indexes.append(r_int)
    # 根据随机数获取随机中心点
    centers = []
    for index in indexes:
        centers.append(input_data.values[index])
    c_ps = []
    for center in centers:
        c_p = 0
        for c_col in center[0: cols]:
            c_p += c_col
        c_ps.append(c_p)
    c_ps.sort()
    # 计算输入数据的参考值
    parameters_data = []
    for i_data in input_data.values:
        p_data = 0
        for i_col in i_data[0:cols]:
            p_data += i_col
        parameters_data.append(p_data)
    # 根据其他对象到每个中心点的距离进行分类
    output_data = k_means_core(input_data, parameters_data, c_ps, cols)

    return output_data


zs_cored_data = pd.read_excel(zs_cored_data_path)
column_names = np.array(zs_cored_data.columns)
column_data = []
for column_name in column_names:
    column_data.append(zs_cored_data[column_name])
column_data = normalize_data(column_data)
normalized_data = []
for col_index in range(0, column_data[0].__len__()):
    row_data = []
    for row_index in range(0, column_data.__len__()):
        row_data.append(column_data[row_index][col_index])
    normalized_data.append(row_data)
column_data_df = pd.DataFrame(normalized_data, columns=zs_cored_data_columns)

# 使用 k-mean 算法进行分类
target_data = k_means(column_data_df, 5, 5)

# 统计聚类类别、聚类个数、聚类中心结果
t_results = []
for t_i in range(0, target_data.__len__()):
    t_d = target_data[t_i]
    r_t_results = []
    for r_t in range(0, t_d[0].__len__()):
        r_t_results.append(0)
    for r_t in range(0, t_d.__len__()):
        r_t_d = t_d[r_t]
        for c_t in range(0, r_t_d.__len__()):
            r_t_results[c_t] += r_t_d[c_t]
    for r_t in range(0, t_d[0].__len__()):
        r_t_results[r_t] = r_t_results[r_t] / t_d.__len__()
    r_t_results.append(t_d.__len__())
    t_results.append(r_t_results)

t_result_columns = np.array(['ZL', 'ZR', 'ZF', 'ZM', 'ZC', 'TYPE'])
t_result_df = pd.DataFrame(t_results, columns=t_result_columns)
t_result_path = '../data/output1.xls'
t_result_df.to_excel(t_result_path)

# 输出原始数据(变换后数据)及其类别，并将结果保存为k-meanfile.xls
k_mean_file_path = '../data/k-meanfile.xls'
k_mean_data = []
for t_i in range(0, target_data.__len__()):
    t_d = target_data[t_i]
    for r_t in range(0, t_d.__len__()):
        r_t_d = list(t_d[r_t])
        r_t_d.append(t_i)
        k_mean_data.append(np.array(r_t_d))

k_mean_df = pd.DataFrame(k_mean_data, columns=np.array(['ZL', 'ZR', 'ZF', 'ZM', 'ZC', 'TYPE']))
k_mean_df.to_excel(k_mean_file_path)
