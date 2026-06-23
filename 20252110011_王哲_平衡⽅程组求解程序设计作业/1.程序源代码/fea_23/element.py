import numpy as np

def get_element_geo(model, e):
    """
    获取第e号单元的几何信息：节点坐标、长度、方向余弦c/s
    :param model: 模型字典
    :param e: 单元编号(0-based)
    :return: x1,y1,x2,y2, L, c, s
    """
    IEN = model["IEN"]
    n1, n2 = IEN[e, 0], IEN[e, 1]
    x1, y1 = model["x"][n1], model["y"][n1]
    x2, y2 = model["x"][n2], model["y"][n2]

    dx = x2 - x1
    dy = y2 - y1
    L = np.hypot(dx, dy)
    if L < 1e-12:
        raise ValueError(f"单元{e}长度为0，模型错误！")
    c = dx / L  # 方向余弦cosθ
    s = dy / L  # 方向余弦sinθ
    return x1, y1, x2, y2, L, c, s

def cal_element_stiffness(model, e):
    """
    计算第e号单元的整体坐标系下单元刚度矩阵Ke
    :param model: 模型字典
    :param e: 单元编号
    :return: Ke (单元刚度矩阵), L, c, s
    """
    nsd = model["nsd"]
    E = model["E"][e]
    A = model["A"][e]
    _, _, _, _, L, c, s = get_element_geo(model, e)
    EA_L = E * A / L

    if nsd == 1:
        # 一维杆单元：2自由度(2节点×1dof)，Ke(2×2)
        Ke = np.array([
            [EA_L, -EA_L],
            [-EA_L, EA_L]
        ], dtype=np.float64)
    elif nsd == 2:
        # 二维桁架单元：4自由度(2节点×2dof)，Ke(4×4)
        Ke = EA_L * np.array([
            [c**2,   c*s,  -c**2,  -c*s],
            [c*s,    s**2, -c*s,   -s**2],
            [-c**2, -c*s,   c**2,   c*s],
            [-c*s,  -s**2,  c*s,    s**2]
        ], dtype=np.float64)
    else:
        raise NotImplementedError("仅支持一维/二维桁架单元")
    return Ke

def cal_stress_force(model, e, de, L, c, s):
    """
    计算单元应力σ、轴力N
    :param model: 模型字典
    :param e: 单元编号
    :param de: 单元局部位移向量
    :param L: 单元长度
    :param c,s: 方向余弦
    :return: sigma(应力), N(轴力)
    """
    E = model["E"][e]
    A = model["A"][e]
    nsd = model["nsd"]

    if nsd == 1:
        coeff = E / L
        sigma = coeff * np.array([-1, 1]) @ de
    elif nsd == 2:
        coeff = E / L
        sigma = coeff * np.array([-c, -s, c, s]) @ de
    else:
        sigma = 0.0

    N = sigma * A  # 轴力 = 应力 × 截面积
    return sigma, N
