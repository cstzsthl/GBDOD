import numpy as np


def GBDOS(data, sigma):
    n, m = data.shape
    delta = np.zeros(m)

    for k in range(m):
        if all(data[:, k] <= 1 + 1e-6):
            delta[k] = sigma

    E = np.zeros(m)
    GBFDens = np.zeros((n, m))

    for k in range(m):
        x = data[:, k]
        rm = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1):
                rm[i, j] = kersim(x[i], x[j], delta[k])
                rm[j, i] = rm[i, j]
        rm_temp = rm
        E[k] = -np.sum((1 / n) * np.log2(np.sum(rm, 1) / n))
        GBFDens[:, k] = np.sum(rm_temp, 1) / n

    W = E / np.sum(E)
    OS = np.zeros(n)
    for i in range(n):
        OS[i] = np.sum((1 - GBFDens[i, :]) * W)
    return OS


# 计算关系矩阵
def kersim(a, x, e):
    if (e == 0):
        if (a == x):
            kersim = 1
        else:
            kersim = 0
    else:
        kersim = max(min((x - a + e) / e, (a - x + e) / e), 0)
    return kersim
