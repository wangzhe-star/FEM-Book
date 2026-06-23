from typing import Tuple, Dict
import numpy as np
import scipy.sparse as sp
from utils import timer_func, vec_norm2

def coo2csr(coo_mat: sp.coo_matrix) -> sp.csr_matrix:
    """COO稀疏矩阵转PARDISO推荐的CSR格式"""
    return coo_mat.tocsr()

@timer_func
def pardiso_solve(A_csr: sp.csr_matrix, b: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    调用Intel MKL PARDISO求解稀疏方程组 A x = b
    :param A_csr: CSR格式稀疏矩阵
    :param b: 右端向量
    :return: 数值解、求解耗时
    """
    x = spsolve(A_csr, b)
    return x

def sparse_post_process(A_csr: sp.csr_matrix, x: np.ndarray, b: np.ndarray) -> Dict:
    """稀疏求解后处理：统计非零元、相对残差、矩阵规模"""
    n = A_csr.shape[0]
    nnz = A_csr.nnz  # 非零元个数
    # 计算残差 r = b - A*x
    r = b - A_csr @ x
    rel_res = vec_norm2(r) / vec_norm2(b) if vec_norm2(b) > 1e-12 else 0.0
    return {
        "matrix_order": n,
        "non_zero_num": nnz,
        "relative_residual": rel_res,
        "sparse_format": "CSR"
    }
