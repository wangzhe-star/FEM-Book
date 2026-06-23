import json
import os
import numpy as np

# ------------------------------
# 基础工具函数
# ------------------------------
def read_json(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在：{file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_txt(file_path, content):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

# ------------------------------
# 任务1：稠密矩阵 LDL^T 求解器
# ------------------------------
def ldlt_factor(K):
    """
    输入对称矩阵 K，返回 L（单位下三角）和 D（对角矩阵）
    若出现非正主元，抛出错误提示
    """
    n = K.shape[0]
    L = np.eye(n, dtype=float)
    D = np.zeros(n, dtype=float)
    
    for i in range(n):
        # 计算 D[i]
        D[i] = K[i, i] - np.dot(L[i, :i] * D[:i], L[i, :i])
        if abs(D[i]) < 1e-12:
            raise ValueError("矩阵非正定，主元为0，无法继续分解")
        # 计算 L[j,i]
        for j in range(i+1, n):
            L[j, i] = (K[j, i] - np.dot(L[j, :i] * D[:i], L[i, :i])) / D[i]
    return L, np.diag(D)

def ldlt_solve(L, D, R):
    """
    求解 L D L^T a = R，包含前代、对角求解和回代
    """
    n = L.shape[0]
    # 前代 Ly = R
    y = np.zeros_like(R, dtype=float)
    for i in range(n):
        y[i] = (R[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    # 对角求解 Dz = y
    z = y / np.diag(D)
    # 回代 L^T a = z
    a = np.zeros_like(z, dtype=float)
    for i in range(n-1, -1, -1):
        a[i] = (z[i] - np.dot(L.T[i, i+1:], a[i+1:])) / L.T[i, i]
    return a

def residual_norm(K, a, R):
    """计算残差 r = R - K@a 及其范数"""
    r = R - K @ a
    return np.linalg.norm(r)

def solve_equilibrium(K, R, method="ldlt"):
    """统一求解接口，使用自己实现的LDLT"""
    if method == "ldlt":
        try:
            L, D = ldlt_factor(K)
            d = ldlt_solve(L, D, R)
            log = {"status": "success"}
        except Exception as e:
            log = {"status": "failed", "error": str(e)}
            d = None
        return d, log
    else:
        raise ValueError(f"不支持的求解方法：{method}")

def calc_relative_error(d, a_exact):
    if d is None or a_exact is None:
        return np.inf
    return np.linalg.norm(d - a_exact) / np.linalg.norm(a_exact)
