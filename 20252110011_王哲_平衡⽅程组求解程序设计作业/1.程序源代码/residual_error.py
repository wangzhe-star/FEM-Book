import numpy as np
from utils import vec_norm2

def calc_relative_residual(r: np.ndarray, R: np.ndarray) -> float:
    """计算相对残差 ||r|| / ||R||"""
    r_norm = vec_norm2(r)
    R_norm = vec_norm2(R)
    if R_norm < 1e-12:
        return 0.0
    return r_norm / R_norm

def calc_relative_error(a_num: np.ndarray, a_exact: np.ndarray) -> float:
    """计算相对误差 ||a_num - a_exact|| / ||a_exact||"""
    diff = a_num - a_exact
    diff_norm = vec_norm2(diff)
    exact_norm = vec_norm2(a_exact)
    if exact_norm < 1e-12:
        return 0.0
    return diff_norm / exact_norm

def calc_condition_num(K: np.ndarray) -> float:
    """计算矩阵2-条件数 cond(K) = ||K||_2 * ||K^{-1}||_2"""
    return np.linalg.cond(K, p=2)

def ill_condition_test(K: np.ndarray, a_exact: np.ndarray):
    """
    病态矩阵测试（任务2）：双精度、4位有效数字、单精度对比
    :param K: 病态系数矩阵
    :param a_exact: 精确解
    """
    R = K @ a_exact
    print("=" * 60)
    print("【病态矩阵误差分析测试】")
    cond = calc_condition_num(K)
    print(f"矩阵2-条件数 cond(K) = {cond:.2e}")

    # 1. 双精度计算
    L_dp, D_dp = ldlt_factor(K)
    a_dp = ldlt_solve(L_dp, D_dp, R)
    r_dp, _ = residual_norm(K, a_dp, R)
    rel_res_dp = calc_relative_residual(r_dp, R)
    rel_err_dp = calc_relative_error(a_dp, a_exact)

    # 2. 四舍五入到4位有效数字
    K_4sig = round_sig_digits(K, 4)
    R_4sig = round_sig_digits(R, 4)
    L_4, D_4 = ldlt_factor(K_4sig)
    a_4sig = ldlt_solve(L_4, D_4, R_4sig)
    r_4, _ = residual_norm(K_4sig, a_4sig, R_4sig)
    rel_res_4 = calc_relative_residual(r_4, R_4sig)
    rel_err_4 = calc_relative_error(a_4sig, a_exact)

    # 3. 单精度(float16)计算
    K_fp16 = K.astype(np.float16)
    R_fp16 = R.astype(np.float16)
    L_fp16, D_fp16 = ldlt_factor(K_fp16.astype(np.float64))
    a_fp16 = ldlt_solve(L_fp16, D_fp16, R_fp16.astype(np.float64))
    r_fp16, _ = residual_norm(K, a_fp16, R)
    rel_res_fp16 = calc_relative_residual(r_fp16, R)
    rel_err_fp16 = calc_relative_error(a_fp16, a_exact)

    # 输出对比结果
    print(f"【双精度】解: {a_dp}, 相对残差: {rel_res_dp:.2e}, 相对误差: {rel_err_dp:.2e}")
    print(f"【4位有效数字】解: {a_4sig}, 相对残差: {rel_res_4:.2e}, 相对误差: {rel_err_4:.2e}")
    print(f"【float16单精度】解: {a_fp16}, 相对残差: {rel_res_fp16:.2e}, 相对误差: {rel_err_fp16:.2e}")
    print("=" * 60)
