import numpy as np

def truss3d_element_stiffness(x1, x2, E, A):
    """
    计算三维杆单元长度、方向余弦、全局刚度矩阵Ke
    :param x1: 节点1坐标 [x1, y1, z1]
    :param x2: 节点2坐标 [x2, y2, z2]
    :param E: 弹性模量 (Pa)
    :param A: 横截面积 (m^2)
    :return: L(长度), dir_cos([cx,cy,cz]), Ke(6×6刚度矩阵)
    """
    # 坐标差
    dx = x2[0] - x1[0]
    dy = x2[1] - x1[1]
    dz = x2[2] - x1[2]

    # 单元长度
    L = np.sqrt(dx**2 + dy**2 + dz**2)

    # 退化单元判断：两节点重合
    if np.isclose(L, 0.0):
        raise ValueError("错误：两个节点坐标重合，单元退化，无法计算！")

    # 方向余弦
    cx = dx / L
    cy = dy / L
    cz = dz / L
    dir_cos = np.array([cx, cy, cz])

    # 构造6×6全局刚度矩阵 Ke
    EA_L = E * A / L
    Ke = np.zeros((6, 6))

    # 按推导公式赋值
    Ke[0, 0] = cx**2
    Ke[0, 1] = cx * cy
    Ke[0, 2] = cx * cz
    Ke[0, 3] = -cx**2
    Ke[0, 4] = -cx * cy
    Ke[0, 5] = -cx * cz

    Ke[1, 0] = cx * cy
    Ke[1, 1] = cy**2
    Ke[1, 2] = cy * cz
    Ke[1, 3] = -cx * cy
    Ke[1, 4] = -cy**2
    Ke[1, 5] = -cy * cz

    Ke[2, 0] = cx * cz
    Ke[2, 1] = cy * cz
    Ke[2, 2] = cz**2
    Ke[2, 3] = -cx * cz
    Ke[2, 4] = -cy * cz
    Ke[2, 5] = -cz**2

    Ke[3, 0] = -cx**2
    Ke[3, 1] = -cx * cy
    Ke[3, 2] = -cx * cz
    Ke[3, 3] = cx**2
    Ke[3, 4] = cx * cy
    Ke[3, 5] = cx * cz

    Ke[4, 0] = -cx * cy
    Ke[4, 1] = -cy**2
    Ke[4, 2] = -cy * cz
    Ke[4, 3] = cx * cy
    Ke[4, 4] = cy**2
    Ke[4, 5] = cy * cz

    Ke[5, 0] = -cx * cz
    Ke[5, 1] = -cy * cz
    Ke[5, 2] = -cz**2
    Ke[5, 3] = cx * cz
    Ke[5, 4] = cy * cz
    Ke[5, 5] = cz**2

    Ke = EA_L * Ke
    return L, dir_cos, Ke


def truss3d_element_stress(x1, x2, E, A, de):
    """
    根据节点位移计算单元应变、应力、轴力
    :param x1: 节点1坐标 [x1,y1,z1]
    :param x2: 节点2坐标 [x2,y2,z2]
    :param E: 弹性模量 (Pa)
    :param A: 横截面积 (m^2)
    :param de: 节点位移列阵 [u1,v1,w1,u2,v2,w2] (m)
    :return: epsilon(应变), sigma(应力,Pa), N(轴力,N)
    """
    L, dir_cos, _ = truss3d_element_stiffness(x1, x2, E, A)
    cx, cy, cz = dir_cos

    # 应变位移矩阵 B
    B = np.array([-cx, -cy, -cz, cx, cy, cz]) / L
    epsilon = B @ np.array(de)

    # 应力、轴力
    sigma = E * epsilon
    N = sigma * A
    return epsilon, sigma, N


def check_matrix_property(Ke):
    """验证刚度矩阵性质：对称性、特征值"""
    print("\n===== 刚度矩阵性质验证 =====")
    # 1. 对称性判断
    is_symmetric = np.allclose(Ke, Ke.T)
    print(f"1. 矩阵是否对称: {is_symmetric}")

    # 2. 计算特征值
    eig_vals = np.linalg.eigvals(Ke)
    print(f"2. 刚度矩阵特征值:\n{np.round(eig_vals, 4)}")
    all_non_neg = np.all(eig_vals >= -1e-8)  # 浮点容错
    print(f"3. 所有特征值是否非负(半正定): {all_non_neg}")

    # 3. 秩与奇异性
    rank = np.linalg.matrix_rank(Ke)
    print(f"4. 矩阵秩: {rank} , 6阶矩阵秩亏 → 矩阵奇异")
    return eig_vals


def rigid_body_test(x1, x2, E, A):
    """刚体位移测试：整体平动，无内力"""
    print("\n===== 刚体位移测试 =====")
    # 整体平移位移：所有节点同方向同大小位移
    de_rigid = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
    eps, sig, n = truss3d_element_stress(x1, x2, E, A, de_rigid)
    print(f"刚体平移下 应变: {eps:.2e}, 应力: {sig:.2e} Pa, 轴力: {n:.2e} N")
    print("结论：刚体位移不产生应变、应力、轴力")


def column_physical_meaning(Ke):
    """任务4：刚度矩阵列的物理意义验证"""
    print("\n===== 刚度矩阵列物理意义验证 =====")
    ndof = 6
    # 选取第4个自由度(j=3,索引从0开始)设为单位位移，其余为0
    j = 3
    de_unit = np.zeros(ndof)
    de_unit[j] = 1.0
    Fe = Ke @ de_unit
    print(f"令第{j+1}个自由度位移=1，其余为0")
    print(f"节点力列阵 Fe = Ke·de:\n{np.round(Fe, 2)}")
    print(f"该结果等价于刚度矩阵第{j+1}列:\n{np.round(Ke[:, j], 2)}")
    print("物理意义：k_ij 表示仅第j个自由度产生单位位移时，第i个自由度所需施加的节点力")


if __name__ == "__main__":
    # ==================== 算例1：沿X轴一维杆单元 ====================
    print("========== 算例1：沿X轴杆单元 ==========")
    x1_1 = [0, 0, 0]
    x2_1 = [2, 0, 0]
    E1 = 200e9    # 200 GPa
    A1 = 1.0e-4   # m^2
    de1 = [0, 0, 0, 1.0e-3, 0, 0]  # 节点位移

    try:
        L1, dir1, Ke1 = truss3d_element_stiffness(x1_1, x2_1, E1, A1)
        eps1, sig1, N1 = truss3d_element_stress(x1_1, x2_1, E1, A1, de1)

        print(f"单元长度 L = {L1:.2f} m")
        print(f"方向余弦 (cx,cy,cz) = {np.round(dir1, 4)}")
        print("6×6单元刚度矩阵 Ke:")
        print(np.round(Ke1, 2))
        print(f"轴向应变 ε = {eps1:.4e}")
        print(f"轴向应力 σ = {sig1/1e6:.2f} MPa")
        print(f"单元轴力 N = {N1:.2f} N")

        # 性质验证
        check_matrix_property(Ke1)
        rigid_body_test(x1_1, x2_1, E1, A1)
    except Exception as e:
        print(e)

    # ==================== 算例2：空间任意方向杆单元 ====================
    print("\n\n========== 算例2：空间任意方向杆单元 ==========")
    x1_2 = [0, 0, 0]
    x2_2 = [1, 2, 2]
    E2 = 210e9    # 210 GPa
    A2 = 2.0e-4   # m^2
    de2 = [0, 0, 0, 1.0e-3, 2.0e-3, 2.0e-3]

    try:
        L2, dir2, Ke2 = truss3d_element_stiffness(x1_2, x2_2, E2, A2)
        eps2, sig2, N2 = truss3d_element_stress(x1_2, x2_2, E2, A2, de2)

        print(f"单元长度 L = {L2:.2f} m")
        print(f"方向余弦 (cx,cy,cz) = {np.round(dir2, 4)}")
        print("6×6单元刚度矩阵 Ke:")
        print(np.round(Ke2, 2))
        print(f"轴向应变 ε = {eps2:.4e}")
        print(f"轴向应力 σ = {sig2/1e6:.2f} MPa")
        print(f"单元轴力 N = {N2:.2f} N")

        # 性质验证
        check_matrix_property(Ke2)
        rigid_body_test(x1_2, x2_2, E2, A2)
        # 刚度矩阵物理意义
        column_physical_meaning(Ke2)
    except Exception as e:
        print(e)
