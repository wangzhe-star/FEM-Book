import numpy as np
import scipy.sparse as sp
from ldlt_solver import ldlt_factor, ldlt_solve, residual_norm
from sparse_solver import coo2csr, pardiso_solve, sparse_post_process

def solve_equilibrium(K_FF, rhs, method="ldlt", **options):
    """
    作业标准接口：求解缩减平衡方程 K_FF d_F = rhs
    :param K_FF: 缩减刚度矩阵（稠密/稀疏）
    :param rhs: 右端向量 f_F - K_EF^T d_E（来自2.3自由度分块）
    :param method: 求解方法："ldlt"(稠密LDL^T) / "pardiso"(PARDISO稀疏求解)
    :param options: 扩展参数（稀疏格式、日志开关等）
    :return: 解向量d_F、求解日志字典
    """
    log = {}
    if method == "ldlt":
        # 稠密LDL^T求解（自研）
        try:
            L, D = ldlt_factor(np.array(K_FF, dtype=np.float64))
            d_F = ldlt_solve(L, D, np.array(rhs, dtype=np.float64))
            r, r_norm = residual_norm(np.array(K_FF), d_F, np.array(rhs))
            log["status"] = "分解成功"
            log["residual_norm"] = r_norm
            log["L_matrix"] = L
            log["D_vector"] = D
        except ValueError as e:
            log["status"] = f"分解失败: {str(e)}"
            d_F = None
        return d_F, log

    elif method == "pardiso":
        # PARDISO稀疏求解（任务3）
        coo_mat = sp.coo_matrix(K_FF)
        csr_mat = coo2csr(coo_mat)
        d_F, solve_time = pardiso_solve(csr_mat, np.array(rhs, dtype=np.float64))
        post_info = sparse_post_process(csr_mat, d_F, np.array(rhs))
        log["status"] = "稀疏求解成功"
        log["solve_time"] = solve_time
        log.update(post_info)
        return d_F, log

    else:
        raise NotImplementedError(f"暂不支持求解方法: {method}")
