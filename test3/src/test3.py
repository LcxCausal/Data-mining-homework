"""
根据下表，采用PCA完成该数据的主成分分析。
x1	x2	x3	x4	x5	x6	x7	x8
40.4	24.7	7.2	6.1	8.3	8.7	2.442	20
25	12.7	11.2	11	12.9	20.2	3.542	9.1
13.2	3.3	3.9	4.3	4.4	5.5	0.578	3.6
22.3	6.7	5.6	3.7	6	7.4	0.176	7.3
34.3	11.8	7.1	7.1	8	8.9	1.726	27.5
35.6	12.5	16.4	16.7	22.8	29.3	3.017	26.6
22	7.8	9.9	10.2	12.6	17.6	0.847	10.6
48.4	13.4	10.9	9.9	10.9	13.9	1.772	17.8
40.6	19.1	19.8	19	29.7	39.6	2.449	35.8
24.8	8	9.8	8.9	11.9	16.2	0.789	13.7
12.5	9.7	4.2	4.2	4.6	6.5	0.874	3.9
1.8	0.6	0.7	0.7	0.8	1.1	0.056	1
32.3	13.9	9.4	8.3	9.8	13.3	2.126	17.1
38.5	9.1	11.3	9.5	12.2	16.4	1.327	11.6
实验要求：
1.	根据上面数据集，给出标准化后的数据，该矩阵均值为0，方差为1；
2.	计算并输出该矩阵的协方差矩阵；
3.	计算并输出协方差矩阵的特征值和特征向量；
4.	计算并输出贡献率和累计贡献率，取累计贡献大于85%的特征向量；
5.	利用该特征向量，计算降维后的数据，并将该数据保存在compress_data.xls中；
提示：
1）	import pandas 和numpy
2）	numpy中covMatrics=cov()求协方差
3）	求特征值、特征向量eigVals, eigVects = np.linalg.eig(np.mat(covMattrics))
4）	注意贡献率、特征值排序需从大到小排序
5）	from sklearn.decomposition import PCA (教育技术专业可直接用PCA中的fit（）建模）
"""

# -*- encoding='uft-8' -*-
import numpy
import xlwt
from sklearn.decomposition import PCA

# 初始化数据
data = [
    [40.4, 24.7, 7.2, 6.1, 8.3, 8.7, 2.442, 20],
    [25, 12.7, 11.2, 11, 12.9, 20.2, 3.542, 9.1],
    [13.2, 3.3, 3.9, 4.3, 4.4, 5.5, 0.578, 3.6],
    [22.3, 6.7, 5.6, 3.7, 6, 7.4, 0.176, 7.3],
    [34.3, 11.8, 7.1, 7.1, 8, 8.9, 1.726, 27.5],
    [35.6, 12.5, 16.4, 16.7, 22.8, 29.3, 3.017, 26.6],
    [22, 7.8, 9.9, 10.2, 12.6, 17.6, 0.847, 10.6],
    [48.4, 13.4, 10.9, 9.9, 10.9, 13.9, 1.772, 17.8],
    [40.6, 19.1, 19.8, 19, 29.7, 39.6, 2.449, 35.8],
    [24.8, 8, 9.8, 8.9, 11.9, 16.2, 0.789, 13.7],
    [12.5, 9.7, 4.2, 4.2, 4.6, 6.5, 0.874, 3.9],
    [1.8, 0.6, 0.7, 0.7, 0.8, 1.1, 0.056, 1],
    [32.3, 13.9, 9.4, 8.3, 9.8, 13.3, 2.126, 17.1],
    [38.5, 9.1, 11.3, 9.5, 12.2, 16.4, 1.327, 11.6]
]

# 1.	根据上面数据集，给出标准化后的数据，该矩阵均值为0，方差为1

# 对原始集求均值
cols_avg = []
cols_count = data[0].__len__()
rows_count = data.__len__()
for col in range(cols_count):
    cols_sum = 0
    for row in range(rows_count):
        cols_sum += data[row][col]
    cols_avg.append(cols_sum / cols_count)

# 原始数据 - 均值
for col in range(cols_count):
    col_avg = cols_avg[col]
    for row in range(rows_count):
        data[row][col] = data[row][col] - col_avg

# 计算标准差
s = []
for col in range(cols_count):
    col_s_sum = 0
    for row in range(rows_count):
        col_s_sum += pow(data[row][col], 2)
    s.append(col_s_sum / cols_count)

# 标准差标准化
for col in range(cols_count):
    for row in range(rows_count):
        data[row][col] = data[row][col] / s[col]

print('标准化后的数据：')
print(data)

# 2.	计算并输出该矩阵的协方差矩阵
pca = PCA()
pca.fit(data)
aa = pca.get_covariance()
aa = aa / aa[0][0]

print('协方差矩阵：')
print(aa)

# 3.	计算并输出协方差矩阵的特征值和特征向量
a, b = numpy.linalg.eig(aa)

print('特征值：')
print(a)

print('特征向量：')
print(b.T)

# 4.	计算并输出贡献率和累计贡献率，取累计贡献大于85%的特征向量
z = []
z_sum = []
z_accumulation = 0
a_sum = 0
for i in range(a.__len__()):
    a_sum += a[i]
for i in range(a.__len__()):
    z.append(a[i] / a_sum)
    z_accumulation += a[i] / a_sum
    z_sum.append(z_accumulation)

for i in range(z.__len__()):
    print('特征值: {0}      贡献值: {1}      累积贡献: {2}'.format(a[i], z[i], z_sum[i]))

print("累计贡献大于85%的特征向量:")
result = []
for i in range(z_sum.__len__()):
    if z_sum[i] < 0.85 or i == 0:
        result.append(b.T[i])
        print('特征向量: {0}'.format(b.T[i]))

# 5.	利用该特征向量，计算降维后的数据，并将该数据保存在compress_data.xls中
r_pca = PCA(n_components=3)
r_data = pca.fit_transform(result)
compress_data_path = '../data/compress_data.xls'
c_d_file = xlwt.Workbook(encoding='utf-8')
c_d_sheet = c_d_file.add_sheet('compress_data')
for i in range(r_data.size):
    c_d_sheet.write(0, i, r_data[0][i])
c_d_file.save(compress_data_path)

