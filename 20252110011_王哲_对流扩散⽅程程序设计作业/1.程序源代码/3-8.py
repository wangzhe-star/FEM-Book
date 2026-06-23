# -*- coding: utf-8 -*-
"""
一维稳态对流扩散方程有限元求解
格式：标准Galerkin、迎风格式(人工扩散)、SUPG/Petrov-Galerkin
作业要求：nel=20，L=1，v=1，对比 Pe=0.1 / Pe=3.0
"""
import numpy as np
import matplotlib.pyplot as plt
from math import isclose
# ===================== 工具函数：SUPG最优alpha =====================
def alpha_supg(Pe):
    """
    计算SUPG最优参数 alpha_opt = coth(Pe) - 1/Pe
    处理Pe趋近于0的除零问题：Pe极小时返回0
    """
    # Pe趋近于0，避免除以0
    if isclose(Pe, 0, abs_tol=1e-10):
        return 0.0
    coth_Pe = np.cosh(Pe) / np.sinh(Pe)
    alpha_opt = coth_Pe - 1.0 / Pe
    return alpha_opt
# ===================== 单元矩阵计算函数 =====================
def element_matrix(kappa, v, le, alpha):
    """
    计算两节点线性单元的对流扩散单元刚度矩阵 Ke (2×2)
    :param kappa: 原始扩散系数
    :param v: 对流速度
    :param le: 单元长度
    :param alpha: 人工扩散系数(0:Galerkin,1:迎风,alpha_opt:SUPG)
    :return Ke: 2×2 单元矩阵
    """
    # 等效扩散系数（引入人工扩散）
    kappa_bar = kappa + alpha * v * le / 2.0
    
    # 扩散项单元矩阵
    K_diff = (kappa_bar / le) * np.array([[1.0, -1.0],
                                         [-1.0, 1.0]])
    # 对流项单元矩阵
    K_conv = (v / 2.0) * np.array([[-1.0, 1.0],
                                   [-1.0, 1.0]])
    # 总单元矩阵
    Ke = K_diff + K_conv
    return Ke
# ===================== 求解对流扩散方程主函数 =====================
def solve_advection_diffusion(nel, L, v, kappa, alpha):
    """
    组装总刚、施加边界条件、求解数值解 + 计算精确解
    :param nel: 单元数量
    :param L: 求解区间长度
    :param v: 对流速度
    :param kappa: 扩散系数
    :param alpha: 人工扩散参数
    :return x: 节点坐标, theta_num: 数值解, theta_ex: 精确解
    """
    # 1. 网格生成
    nn = nel + 1                # 总节点数
    le = L / nel                # 单元长度
    x = np.linspace(0, L, nn)   # 节点坐标
    
    # 2. 初始化总体刚度矩阵 & 右端向量
    K_global = np.zeros((nn, nn))
    F = np.zeros(nn)
    
    # 3. 遍历所有单元，组装总体矩阵
    for e in range(nel):
        # 单元局部节点 -> 全局节点编号
        node1 = e
        node2 = e + 1
        # 计算当前单元矩阵
        Ke = element_matrix(kappa, v, le, alpha)
        # 叠加到总刚矩阵
        K_global[node1:node2+1, node1:node2+1] += Ke
    
    # 4. 施加狄利克雷边界条件
    # 左边界 x=0: theta=0 (第0号节点)
    K_global[0, :] = 0.0
    K_global[0, 0] = 1.0
    F[0] = 0.0
    
    # 右边界 x=L: theta=1 (最后一个节点)
    K_global[-1, :] = 0.0
    K_global[-1, -1] = 1.0
    F[-1] = 1.0
    
    # 5. 求解线性方程组 K*theta = F
    theta_num = np.linalg.solve(K_global, F)
    
    # 6. 计算精确解（使用expm1避免指数溢出）
    arg = v * x / kappa
    arg_total = v * L / kappa
    theta_ex = np.expm1(arg) / np.expm1(arg_total)
    
    return x, theta_num, theta_ex, K_global
# ===================== 辅助函数：计算最大节点误差 =====================
def calc_max_error(theta_num, theta_ex):
    """计算数值解与精确解的最大绝对误差"""
    err = np.abs(theta_num - theta_ex)
    return np.max(err)
# ===================== 主程序入口 =====================
if __name__ == "__main__":
    # 全局固定参数（作业要求）
    L = 1.0
    nel = 20
    v = 1.0
    Pe_list = [0.1, 3.0]  # 两组Peclet数工况
    alpha_name = {0: "标准Galerkin", 1: "迎风格式", "supg": "SUPG"}
    
    # ========== 字体修改为：微软雅黑、宋体 ==========
    plt.rcParams["font.family"] = ["Microsoft YaHei", "SimSun"]
    plt.rcParams["axes.unicode_minus"] = False
    
    # ============= 任务1 & 任务2：遍历两组Pe工况计算求解 =============
    error_table = []  # 误差统计表
    K_galerkin_Pe3 = None  # 保存Pe=3.0时标准Galerkin总矩阵（用于矩阵分析）
    
    for Pe in Pe_list:
        le = L / nel
        # 由单元Pe数反求扩散系数 kappa: Pe = v*le/(2*kappa)
        kappa = (v * le) / (2 * Pe)
        print(f"========== 当前工况：单元Pe = {Pe:.2f} ==========")
        print(f"单元长度 le = {le:.4f}, 扩散系数 kappa = {kappa:.8f}")
        
        # 1. 标准Galerkin: alpha=0
        x_gal, theta_gal, theta_ex, _ = solve_advection_diffusion(nel, L, v, kappa, alpha=0)
        err_gal = calc_max_error(theta_gal, theta_ex)
        
        # 2. 迎风格式: alpha=1
        x_up, theta_up, _, _ = solve_advection_diffusion(nel, L, v, kappa, alpha=1)
        err_up = calc_max_error(theta_up, theta_ex)
        
        # 3. SUPG格式: alpha=alpha_opt
        alpha_opt = alpha_supg(Pe)
        print(f"SUPG最优参数 alpha_opt = {alpha_opt:.6f}")
        x_supg, theta_supg, _, K_temp = solve_advection_diffusion(nel, L, v, kappa, alpha=alpha_opt)
        err_supg = calc_max_error(theta_supg, theta_ex)
        
        # 保存Pe=3.0的Galerkin总矩阵，用于后续矩阵分析
        if isclose(Pe, 3.0):
            K_galerkin_Pe3 = K_temp
        
        # 记录误差
        error_table.append([Pe, err_gal, err_up, err_supg])
        print(f"标准Galerkin 最大误差: {err_gal:.8f}")
        print(f"迎风格式     最大误差: {err_up:.8f}")
        print(f"SUPG格式     最大误差: {err_supg:.8f}\n")
    
    # ========== 【修改部分】输出规整的三种格式最大节点绝对误差汇总表，匹配作业报告要求 ==========
    print("="*80)
    print("                三种格式最大节点绝对误差汇总表")
    print("="*80)
    header = f"{'单元Pe数':<12}{'标准Galerkin误差':<18}{'迎风格式误差':<18}{'SUPG稳定化格式误差':<18}"
    print(header)
    print("-"*80)
    for row in error_table:
        pe_val, e_gal, e_up, e_supg = row
        line = f"{pe_val:<12.1f}{e_gal:<18.8f}{e_up:<18.8f}{e_supg:<18.8f}"
        print(line)
    print("="*80 + "\n")
    
    # ============= 任务4：Pe=3.0 标准Galerkin矩阵性质分析 =============
    print("==================== Pe=3.0 标准Galerkin总矩阵分析 ====================")
    K = K_galerkin_Pe3
    nn = K.shape[0]
    print(f"总矩阵维度: {nn} × {nn}")
    
    # 1. 判断矩阵是否对称
    is_symmetric = np.allclose(K, K.T, atol=1e-10)
    print(f"矩阵是否对称: {'是' if is_symmetric else '否'}")
    
    # 2. 判断矩阵是否正定（特征值全大于0）
    eig_vals = np.linalg.eigvals(K)
    min_eig = np.min(eig_vals)
    all_positive = np.all(eig_vals > -1e-10)
    print(f"矩阵最小特征值: {min_eig:.6e}")
    print(f"矩阵是否正定: {'是' if all_positive else '否'}")
    print("结论：对流项引入反对称分量，矩阵非对称；对流占优时出现负特征值，矩阵非正定。")
    
    # ============= 附加题：网格加密收敛分析 nel=10,20,40,80 =============
    print("\n==================== 附加题：网格加密收敛分析 ====================")
    nel_list = [10, 20, 40, 80]
    Pe_fixed = 3.0  # 固定对流占优工况Pe=3.0
    err_gal_grid = []
    err_supg_grid = []
    
    for nel_curr in nel_list:
        le_curr = L / nel_curr
        kappa_curr = (v * le_curr) / (2 * Pe_fixed)
        # 标准Galerkin
        _, th_gal, th_ex, _ = solve_advection_diffusion(nel_curr, L, v, kappa_curr, alpha=0)
        e_gal = calc_max_error(th_gal, th_ex)
        # SUPG
        alpha_opt_curr = alpha_supg(Pe_fixed)
        _, th_supg, _, _ = solve_advection_diffusion(nel_curr, L, v, kappa_curr, alpha=alpha_opt_curr)
        e_supg = calc_max_error(th_supg, th_ex)
        
        err_gal_grid.append(e_gal)
        err_supg_grid.append(e_supg)
        print(f"单元数 nel={nel_curr:3d} | Galerkin误差={e_gal:.8e} | SUPG误差={e_supg:.8e}")
    
    # 绘制网格收敛误差曲线
    plt.figure(figsize=(10, 6))
    plt.plot(nel_list, err_gal_grid, 'ro-', linewidth=1.5, markersize=6, label='标准Galerkin')
    plt.plot(nel_list, err_supg_grid, 'bo-', linewidth=1.5, markersize=6, label='SUPG格式')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("单元数量 nel (对数坐标)", fontsize=12)
    plt.ylabel("最大绝对误差 (对数坐标)", fontsize=12)
    plt.title(f"网格加密收敛曲线 (固定单元Pe={Pe_fixed})", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig("grid_convergence.png", dpi=300)
    plt.show()
