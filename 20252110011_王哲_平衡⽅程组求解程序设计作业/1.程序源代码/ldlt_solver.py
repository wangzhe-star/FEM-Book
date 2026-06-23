from typing import Tuple
import numpy as np

def ldlt_factor(K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    对称矩阵 LDL^T 分解：K = L @ D @ L^T
    :param K: 输入对称方阵（稠密矩阵），下标0开始
    :return: L(单位下三角矩阵), D(对角矩阵，一维数组存储)
    :raises ValueError: 检测到非正主元（矩阵非正定/奇异）
    """
    n = K.shape[0]
    if K.shape[1] != n:
        raise ValueError("输入必须为方阵")
    # 校验矩阵对称性（浮点容差1e-10）
    if not np.allclose(K, K.T, atol=1e-10):
        raise ValueError("输入矩阵非对称，无法进行LDL^T分解")

    L = np.eye(n, dtype=np.float64)  # 单位下三角阵
    D = np.zeros(n, dtype=np.float64) # 对角元，一维存储

    for j in range(n):
        # 步骤1：计算D[j] = K[j,j] - sum(L[j,k]^2 * D[k])  k=0~j-1
        sum_d = 0.0
        for k in range(j):
            sum_d += L[j, k] ** 2 * D[k]
        D[j] = K[j, j] - sum_d

        # 检测非正主元（作业强制要求）
        if D[j] <= 1e-12:  # 浮点零阈值
            raise ValueError(f"LDL^T分解失败：第{j}个主元 D[{j}] = {D[j]:.6e}，矩阵非正定或存在零主元")

        # 步骤2：计算L[i,j]  i = j+1 ~ n-1
        for i in range(j + 1, n):
            sum_l = 0.0
            for k in range(j):
                sum_l += L[i, k] * L[j, k] * D[k]
            L[i, j] = (K[i, j] - sum_l) / D[j]
    return L, D

def ldlt_solve(L: np.ndarray, D: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    基于LDL^T分解求解线性方程组 L D L^T a = R
    三步流程：前代 → 对角求解 → 回代
    :param L: 单位下三角矩阵
    :param D: 对角元数组
    :param R: 右端向量
    :return: 解向量 a
    """
    n = L.shape[0]
    # ========== 第一步：前代 L y = R （单位下三角） ==========
    y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        y[i] = R[i]
        for k in range(i):
            y[i] -= L[i, k] * y[k]

    # ========== 第二步：对角求解 D z = y ==========
    z = y / D

    # ========== 第三步：回代 L^T a = z （单位上三角） ==========
    a = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        a[i] = z[i]
        for k in range(i + 1, n):
            a[i] -= L[k, i] * a[k]
    return a

def residual_norm(K: np.ndarray, a: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    计算残差向量 r = R - K*a 及残差2-范数（作业要求函数）
    :param K: 系数矩阵
    :param a: 数值解
    :param R: 右端向量
    :return: 残差向量、残差2-范数
    """
    r = R - K @ a
    r_norm = np.linalg.norm(r, ord=2)
    return r, r_norm
