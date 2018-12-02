"""
下图是亚洲15只球队在2005-2010年间大型比赛的战绩（澳大利亚未收录）。
数据做了如下预处理：对于世界杯，进入决赛则取其最终排名，没有进入决赛圈的，打入预赛十强赛的赋予40，
预选赛小组未出线的赋予50。对于亚洲杯，前四名取其排名，八强赋予5，十六强赋予9，预选赛没有出线的赋予17。
"""

from random import randint
import pandas as pd


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
            for o_col in o_data[1:(cols + 1)]:
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


# 实现 k_means 算法
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
        for c_col in center[1: (cols + 1)]:
            c_p += c_col
        c_ps.append(c_p)
    c_ps.sort()
    # 计算输入数据的参考值
    parameters_data = []
    for i_data in input_data.values:
        p_data = 0
        for i_col in i_data[1:(cols + 1)]:
            p_data += i_col
        parameters_data.append(p_data)
    # 根据其他对象到每个中心点的距离进行分类
    output_data = k_means_core(input_data, parameters_data, c_ps, cols)

    return output_data


def convert_to_chinese(num):
    if num.__eq__(1):
        return '一'
    elif num.__eq__(2):
        return '二'
    elif num.__eq__(3):
        return '三'
    else:
        return '不入'


# 通过 pandas 读取 ../data/data.xls 文件的数据
source_data_path = "../data/data.xls"
source_data = pd.read_excel(source_data_path)

# 当 k=3 时， 对数据进行聚类划分
target_data = k_means(source_data, 3, 3)

# 打印聚类结果
for i_t in range(0, target_data.__len__()):
    print('{0}流水平国家队:'.format(convert_to_chinese(i_t + 1)))
    print(target_data[i_t])
print()

# 判断中国属于哪个水平
country = '中国'
for t_i in range(0, target_data.__len__()):
    for t_d in target_data[t_i]:
        if t_d[0] == country:
            print('中国对目前属于{0}流水平'.format(convert_to_chinese(t_i + 1)))
