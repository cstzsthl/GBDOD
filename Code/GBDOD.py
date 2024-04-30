from sklearn.metrics import roc_auc_score
from MGB import getGranularball
from GBDOS import GBDOS
import numpy as np
from scipy.io import loadmat
import time

def get_fea_type(data):
    n, m = data.shape
    fea_type = np.zeros(m)
    ID = (data.min(axis=0) == 1)
    fea_type[ID] = 1

    return fea_type


if __name__ == "__main__":
    key = ["test"]
    load_data = loadmat(r"E:\datasets\Numerical\\" + key + ".mat")
    data = load_data['trandata']
    n, m = data.shape
    label = data[:, m - 1]
    trandata = data[:, :m - 1]

    # 0 represents numerical attribute，1 represents nominal attribute
    fea_type = get_fea_type(trandata)

    continuous_features = np.where(fea_type == 0)[0]
    continuous_features.tolist()

    centers, gb_list = getGranularball(data[:, :m - 1], fea_type)

    start_time = time.time()
    index = []
    for gb in gb_list:
        index.append(gb[:, m - 1])

    index_len = len(index)
    out_scores = np.zeros(n)

    sigma = 1

    out_scores_centers = GBDOS(centers, sigma)
    for i in range(0, index_len):
        for j in index[i]:
            out_scores[int(j)] = out_scores_centers[i]

    AUC = roc_auc_score(label, out_scores)

