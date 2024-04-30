from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import mode, entropy
import numpy as np
from sklearn.cluster import k_means
from kmodes.kprototypes import KPrototypes
from kmodes.kmodes import KModes

import warnings

warnings.filterwarnings('ignore')


def get_radius_new(GB, fea_type, numerical, categorical, gamma):
    n, m = GB.shape
    center = get_center(GB[:, :m - 1], fea_type)
    diffMat = get_diffMat(GB[:, :m - 1], center, fea_type)
    sqDiffMat = diffMat ** 2
    numerical_diffMat, categorical_diffMat = _split_num_cat(sqDiffMat, numerical, categorical)
    distances = numerical_diffMat.sum(axis=1) ** 0.5 + gamma * categorical_diffMat.sum(axis=1) ** 0.5

    radius = max(distances)
    return radius


def get_center(GB, fea_type):
    n, m = GB.shape

    center = np.zeros(m)

    for col in range(0, m):
        if fea_type[col] == 0:  # 连续数据，计算均值
            center[col] = GB[:, col].mean()
        elif fea_type[col] == 1:  # 标称数据，计算众数
            center[col] = mode(GB[:, col], axis=None, keepdims=True).mode[0]

    return center


def get_diffMat(GB, center, fea_type):
    diffMat = np.zeros_like(GB)
    for col, ftype in enumerate(fea_type):
        if ftype == 0:
            diffMat[:, col] = GB[:, col] - center[col]
        elif ftype == 1:
            diffMat[:, col] = np.where(GB[:, col] == center[col], 0, 1)

    return diffMat


def get_density_new(GB, fea_type, numerical, categorical, gamma):
    n, m = GB.shape
    center = get_center(GB[:, :m - 1], fea_type)
    diffMat = get_diffMat(GB[:, :m - 1], center, fea_type)
    sqDiffMat = diffMat ** 2
    numerical_diffMat, categorical_diffMat = _split_num_cat(sqDiffMat, numerical, categorical)
    distances = numerical_diffMat.sum(axis=1) ** 0.5 + gamma * categorical_diffMat.sum(axis=1) ** 0.5

    sum_of_radius = 0

    for i in distances:
        sum_of_radius += i

    if sum_of_radius != 0:
        density = sum_of_radius / n
    else:
        density = n

    return density


def split_ball_1(GB, cat_features, gamma):
    n, m = GB.shape
    if m - 1 > len(cat_features) > 0:  # Mixed
        kproto = KPrototypes(n_clusters=2, init='Cao', n_init=20, max_iter=150, verbose=0,
                             gamma=gamma)  # 这里的gamma值用属性的信息熵来定义
        clusters = kproto.fit_predict(GB[:, :m - 1], categorical=cat_features)
    elif len(cat_features) == 0:  # numerical
        clusters = k_means(X=GB[:, :m - 1], init='k-means++', n_clusters=2)[1]
    elif len(cat_features) == m - 1:  # nominal
        k_modes_model = KModes(n_clusters=2, init='Huang', n_init=20, max_iter=150, verbose=0)
        clusters = k_modes_model.fit_predict(GB[:, :m - 1])

    ball_a = GB[clusters == 0, :]
    ball_b = GB[clusters == 1, :]

    return [ball_a, ball_b]


def division_circle(GB_list, fea_type, numerical, categorical, gamma, data_num):
    GB_list_temp = []
    for GB in GB_list:
        if len(GB) >= 8:  # 这里的最小划分粒球应该设置为多少？
            ball_A, ball_B = split_ball_1(GB, categorical, gamma)
            density_origin = get_density_new(GB, fea_type, numerical, categorical, gamma)
            density_ball_A = get_density_new(ball_A, fea_type, numerical, categorical, gamma)
            density_ball_B = get_density_new(ball_B, fea_type, numerical, categorical, gamma)
            weight_A = len(ball_A) / len(GB)
            weight_B = len(ball_B) / len(GB)
            weighted_density = weight_A * density_ball_A + weight_B * density_ball_B
            if density_origin >= weighted_density:
                GB_list_temp.extend([ball_A, ball_B])
            else:
                GB_list_temp.append(GB)
        else:
            GB_list_temp.append(GB)

    return GB_list_temp


def normalize(GB_list, radius_to_split, fea_type, continuous_features, cat_features, gamma):
    gb_list_temp = []
    for gb in GB_list:
        if len(gb) < 2:
            gb_list_temp.append(gb)
        else:
            ball_A, ball_B = split_ball_1(gb, cat_features, gamma)
            if get_radius_new(gb, fea_type, continuous_features, cat_features, gamma) <= 2 * radius_to_split:
                gb_list_temp.append(gb)
            else:
                gb_list_temp.extend([ball_A, ball_B])

    return gb_list_temp


def _split_num_cat(data, numerical, categorical):
    data_numerical = np.asanyarray(data[:, numerical])
    data_categorical = np.asanyarray(data[:, categorical])
    return data_numerical, data_categorical


def column_entropy(column):
    unique_values, counts = np.unique(column, return_counts=True)
    probabilities = counts / np.sum(counts)
    return entropy(probabilities, base=2)


def getGranularball(data, fea_type):
    np.set_printoptions(suppress=True)
    data_num, fea_num = data.shape
    continuous_features = np.where(np.array(fea_type) == 0)[0]
    cat_features = np.where(np.array(fea_type) == 1)[0]
    cat_features = cat_features.tolist()

    data_numerical, data_categorical = _split_num_cat(data, continuous_features, cat_features)
    if len(cat_features) != 0 and len(continuous_features) != 0:
        std_numerical = np.mean(data_numerical.std(axis=0))
        cat_entrophy = np.mean(np.apply_along_axis(column_entropy, axis=0, arr=data_categorical))
        gamma = cat_entrophy / (10 * std_numerical)
    else:
        gamma = 1

    scaler = MinMaxScaler(feature_range=(0, 1))
    if sum(fea_type) != fea_num:
        data[:, continuous_features] = scaler.fit_transform(data[:, continuous_features])
    index = np.array(range(data_num)).reshape(data_num, 1)
    data = np.hstack((data, index))

    gb_list_temp = [data]

    while 1:
        ball_number_old = len(gb_list_temp)
        gb_list_temp = division_circle(gb_list_temp, fea_type, continuous_features, cat_features, gamma, data_num)
        ball_number_new = len(gb_list_temp)
        if ball_number_new == ball_number_old:
            break

    radius = []
    for gb in gb_list_temp:
        if len(gb) >= 2:
            radius.append(get_radius_new(gb, fea_type, continuous_features, cat_features, gamma))
    radius_median = np.median(radius)
    radius_mean = np.mean(radius)
    radius_detect = max(radius_median, radius_mean)
    while 1:
        ball_number_old = len(gb_list_temp)
        gb_list_temp = normalize(gb_list_temp, radius_detect, fea_type, continuous_features, cat_features, gamma)
        ball_number_new = len(gb_list_temp)
        if ball_number_new == ball_number_old:
            break

    gb_list_final = gb_list_temp
    gb_centers = []

    for gb in gb_list_final:
        gb_centers.append(get_center(gb[:, :fea_num], fea_type))

    centers = np.vstack(gb_centers)

    return centers, gb_list_final
