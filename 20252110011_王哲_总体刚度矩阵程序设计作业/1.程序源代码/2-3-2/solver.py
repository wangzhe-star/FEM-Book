import numpy as np

def judge_singular(mat, tol=1e-10):
    """判断方阵是否奇异：秩 < 阶数 则奇异"""
    rank = np.linalg.matrix_rank(mat, tol=tol)
    return rank < mat.shape[0]

def solve_reduction(model, K, LM):
    """
    缩减法处理位移边界条件，求解整体位移、约束反力
    :param model: 模型字典
    :param K: 总体刚度矩阵
    :param LM: 对号矩阵
    :return: d(整体位移), R(约束反力)
    """
    total_dof = model["total_dof"]
    fixed_dof = model["fixed_dof"]
    fixed_val = model["fixed_val"]
    F = model["F"].copy()

    # 1. 划分自由度：E=约束自由度(已知位移)，F=自由自由度(未知位移)
    all_dof = np.arange(total_dof)
    free_dof = np.setdiff1d(all_dof, fixed_dof)  # 自由自由度集合F
    E = fixed_dof
    F_set = free_dof

    # 2. 分块矩阵提取
    K_FF = K[np.ix_(F_set, F_set)]
    K_EF = K[np.ix_(E, F_set)]
    K_FE = K[np.ix_(F_set, E)]
    K_EE = K[np.ix_(E, E)]

    f_F = F[F_set]
    d_E = fixed_val

    # 3. 求解核心方程: K_FF * d_F = f_F - K_EF^T @ d_E
    rhs = f_F - K_EF.T @ d_E
    d_F = np.linalg.solve(K_FF, rhs)

    # 4. 重构完整整体位移向量
    d = np.zeros(total_dof, dtype=np.float64)
    d[F_set] = d_F
    d[E] = d_E

    # 5. 计算约束自由度上的约束反力 R
    R = K_EF @ d_F + K_EE @ d_E - F[E]

    return d, R, K_FF
