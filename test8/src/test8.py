# -*- encoding='utf-8' -*-

# 1)	使用from sklearn  import  datasets中make_circle和make_blobs随机生成环状和球状数据。
# （环状数据生成参考：https://blog.csdn.net/dataningwei/article/details/53649330）
# 2)	分别采用k-means和DBSCAN算法聚类生成数据，并将聚类结果可视化。
# 3)	通过调整ε 和min_Pts观察聚类效果。

from random import randint
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import numpy as np
import math


# 实现 k-means 聚类算法
def k_means(input_data, k):
    # 生成 k 个随机数
    indexes = []
    while not indexes.__len__().__eq__(k):
        r_int = randint(0, input_data.shape[0])
        if not indexes.__contains__(r_int):
            indexes.append(r_int)

    # 根据随机数获取随机中心点
    centers = []
    for index in indexes:
        centers.append(input_data[index])

    # 根据其他对象到每个中心点的距离进行聚类
    output_data = k_means_core(input_data, centers)

    return output_data


# 根据其他对象到每个中心点的距离进行聚类
def k_means_core(input_data, centers):
    # 初始化聚类个数
    output_data = []
    for i in range(0, centers.__len__()):
        output_data.append([])

    # 对输入的数据进行聚类
    for i in range(0, input_data.__len__()):

        # 计算该点与第一个聚类中心点的距离
        min_distance_index = 0
        min_distance = get_distance_from_two_points(centers[0], input_data[i])

        # 计算该点与其他聚类中心点的距离
        for j in range(1, centers.__len__()):
            distance = get_distance_from_two_points(centers[j], input_data[i])

            # 判断与该点距离最近的一个中心点
            if min_distance > distance:
                min_distance = distance
                min_distance_index = j

        # 聚类
        output_data[min_distance_index].append(input_data[i])

    # 计算新簇的中心值
    new_center = []
    for data in output_data:
        sum_x = 0
        sum_y = 0
        for item in data:
            sum_x += item[0]
            sum_y += item[1]
        new_center.append([sum_x / data.__len__(), sum_y / data.__len__()])

    # 判断新簇的中心点是否稳定，若不稳定则根据新簇的中心点重新进行聚类
    is_regroup = False
    for i in range(0, centers.__len__()):
        if not new_center[i].__eq__(centers[i]):
            is_regroup = True
            break

    if is_regroup:
        output_data = k_means_core(input_data, new_center)

    return output_data


# 计算两点之间的距离
def get_distance_from_two_points(begin_point, end_point):
    w = math.fabs(begin_point[0] - end_point[0])
    h = math.fabs(begin_point[1] - end_point[1])
    return math.sqrt(math.pow(w, 2) + math.pow(h, 2))


# 实现 DBSCAN 聚类算法
def density_based_spatial_clustering_of_applications_with_noise(input_data, r, min_pts):
    output_data = []
    output_data_indexes = []

    # 标记所有对象为 unvisited
    input_data_visited = []
    for i in range(0, input_data.__len__()):
        input_data_visited.append(False)

    while check_existed_unvisited_item(input_data_visited):

        # 随机选择一个 unvisited 的对象并标记为 True
        unvisited_data, unvisited_index = get_unvisited_item_by_random(input_data, input_data_visited)
        input_data_visited[unvisited_index] = True

        # 检查该对象的 r 邻域内是否至少有 min_pts 个对象
        if check_r_range_existed_min_pts(unvisited_data, input_data, r, min_pts):

            # 创建一个新簇
            new_category_data = []
            output_data.append(new_category_data)

            # 获取该对象 r 邻域内的所有对象集合
            r_data, r_data_indexes = get_r_range_data(unvisited_data, input_data, r)

            # 选择邻域中符合条件的对象加入当前簇
            r_data_length = r_data.__len__()
            j = 0
            while j < r_data_length:
                r_data_index = r_data_indexes[j]
                if not input_data_visited[r_data_index]:
                    input_data_visited[r_data_index] = True

                    # 如果 r_data[j] 的邻域至少有 min_pts 个对象，把这些对象添加到 r_data 集合
                    if check_r_range_existed_min_pts(r_data[j], input_data, r, min_pts):
                        new_r_data, new_r_data_indexes = get_r_range_data(r_data[j], input_data, r)
                        for k in range(0, new_r_data_indexes.__len__()):
                            new_r_data_index = new_r_data_indexes[k]
                            if not r_data_indexes.__contains__(new_r_data_index):
                                r_data_indexes.append(new_r_data_index)
                                r_data.append(new_r_data[k])
                    r_data_length = r_data.__len__()

                    # 如果 r_data[j] 还不是任何簇的成员， 将其添加到当前簇
                    if not output_data_indexes.__contains__(r_data_index):
                        output_data_indexes.append(r_data_index)
                        new_category_data.append(r_data[j])
                j += 1

    return output_data


# 获取该对象 r 邻域内的所有对象集合
def get_r_range_data(unvisited_data, input_data, r):
    r_data = []
    r_data_indexes = []
    for i in range(0, input_data.__len__()):
        data = input_data[i]
        distance = get_distance_from_two_points(unvisited_data, data)
        if distance <= r:
            r_data.append(data)
            r_data_indexes.append(i)
    return r_data, r_data_indexes


# 检查 unvisited 对象的 r 邻域是否至少有 min_pts 个对象
def check_r_range_existed_min_pts(unvisited_data, input_data, r, min_pts):
    pts = 0
    for data in input_data:
        distance = get_distance_from_two_points(unvisited_data, data)
        if distance <= r:
            pts += 1
    return pts - 1 >= min_pts


# 随机返回一个 unvisited 的对象
def get_unvisited_item_by_random(input_data, input_data_visited):
    unvisited_data = []
    unvisited_data_indexes = []
    for i in range(0, input_data_visited.__len__()):
        if not input_data_visited[i]:
            unvisited_data.append(input_data[i])
            unvisited_data_indexes.append(i)
    unvisited_index = randint(0, unvisited_data.__len__() - 1)
    return unvisited_data[unvisited_index], unvisited_data_indexes[unvisited_index]


# 检查是否存在 unvisited 的元素
def check_existed_unvisited_item(input_data):
    is_existed = False
    for data in input_data:
        if not data:
            is_existed = True
            break
    return is_existed


plt.figure(figsize=(8, 8))
plt.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)

# 生成球状数据
blobs_data, blobs_data_category = make_blobs(n_samples=1000, n_features=2, centers=5)

# 生成环状数据
circles_data, circles_data_category = make_circles(n_samples=1000, factor=0.6, noise=0.02)

# 绘制生成的球状数据
plt.subplot(3, 2, 1)
plt.title("blobs data")
plt.scatter(blobs_data[:, 0], blobs_data[:, 1], marker='o', c=blobs_data_category)

# 绘制生成的环状数据
plt.subplot(3, 2, 2)
plt.title("circles data")
plt.scatter(circles_data[:, 0], circles_data[:, 1], marker='o', c=circles_data_category)

# 通过 k-means 聚类算法对球状数据进行聚类
k_means_output_blobs_data = k_means(blobs_data, 3)

# 对球状数据的聚类结果进行整理
k_means_blobs_data = []
k_means_blobs_data_category = []
for c_i in range(0, k_means_output_blobs_data.__len__()):
    category_data = k_means_output_blobs_data[c_i]
    for row_data in category_data:
        k_means_blobs_data.append(row_data)
        k_means_blobs_data_category.append(c_i)
k_means_blobs_data_np_array = np.array(k_means_blobs_data)
k_means_blobs_data_category_np_array = np.array(k_means_blobs_data_category)

# 绘制 k-means 聚类后的球状数据
plt.subplot(3, 2, 3)
plt.title("k-means blobs data")
plt.scatter(k_means_blobs_data_np_array[:, 0], k_means_blobs_data_np_array[:, 1],
            marker='o', c=k_means_blobs_data_category_np_array)

# 通过 k-means 聚类算法对环状数据进行聚类
k_means_output_circles_data = k_means(circles_data, 3)

# 对环状数据的聚类结果进行整理
k_means_circles_data = []
k_means_circles_data_category = []
for c_i in range(0, k_means_output_circles_data.__len__()):
    category_data = k_means_output_circles_data[c_i]
    for row_data in category_data:
        k_means_circles_data.append(row_data)
        k_means_circles_data_category.append(c_i)
k_means_circles_data_np_array = np.array(k_means_circles_data)
k_means_circles_data_category_np_array = np.array(k_means_circles_data_category)

# 绘制 k-means 聚类后的环状数据
plt.subplot(3, 2, 4)
plt.title("k-means circles data")
plt.scatter(k_means_circles_data_np_array[:, 0], k_means_circles_data_np_array[:, 1],
            marker='o', c=k_means_circles_data_category_np_array)

# 通过 DBSCAN 聚类算法对球状数据进行聚类
dbscan_output_blobs_data = density_based_spatial_clustering_of_applications_with_noise(blobs_data, 1, 3)

# 对球状数据的聚类结果进行整理
dbscan_blobs_data = []
dbscan_blobs_data_category = []
for c_i in range(0, dbscan_output_blobs_data.__len__()):
    category_data = dbscan_output_blobs_data[c_i]
    for row_data in category_data:
        dbscan_blobs_data.append(row_data)
        dbscan_blobs_data_category.append(c_i)
dbscan_blobs_data_np_array = np.array(dbscan_blobs_data)
dbscan_blobs_data_category_np_array = np.array(dbscan_blobs_data_category)

# 绘制 DBSCAN 聚类后的球状数据
plt.subplot(3, 2, 5)
plt.title("dbscan blobs data")
plt.scatter(dbscan_blobs_data_np_array[:, 0], dbscan_blobs_data_np_array[:, 1],
            marker='o', c=dbscan_blobs_data_category_np_array)

# 通过 DBSCAN 聚类算法对环状数据进行聚类
dbscan_output_circles_data = density_based_spatial_clustering_of_applications_with_noise(circles_data, 0.08, 3)

# 对环状数据的聚类结果进行整理
dbscan_circles_data = []
dbscan_circles_data_category = []
for c_i in range(0, dbscan_output_circles_data.__len__()):
    category_data = dbscan_output_circles_data[c_i]
    for row_data in category_data:
        dbscan_circles_data.append(row_data)
        dbscan_circles_data_category.append(c_i)
dbscan_circles_data_np_array = np.array(dbscan_circles_data)
dbscan_circles_data_category_np_array = np.array(dbscan_circles_data_category)

# 绘制 DBSCAN 聚类后的环状数据
plt.subplot(3, 2, 6)
plt.title("dbscan circles data")
plt.scatter(dbscan_circles_data_np_array[:, 0], dbscan_circles_data_np_array[:, 1],
            marker='o', c=dbscan_circles_data_category_np_array)

# 展示聚类后的球状数据和环状数据以确保数据成功生成
plt.show()
