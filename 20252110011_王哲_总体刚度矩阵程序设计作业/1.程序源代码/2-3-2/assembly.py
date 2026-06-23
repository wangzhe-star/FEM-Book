import numpy as np
from element import cal_element_stiffness

def gen_LM(model):
    """
    生成对号矩阵 LM (Location Matrix)
    LM维度：(单单元总自由度, 总单元数)
    :param model: 模型字典
    :return: LM 矩阵
    """
    nel = model["nel"]
    nen = model["nen"]
    ndof_node = model["ndof_per_node"]
    IEN = model["IEN"]
    elem_dof = nen * ndof_node  # 单个单元自由度总数
    LM = np.zeros((elem_dof, nel), dtype=int)

    for e in range(nel):
        # 当前单元的两个节点
        n1, n2 = IEN[e, 0], IEN[e, 1]
        # 节点1对应的全局自由度起始编号
        dof1_start = n1 * ndof_node
        # 节点2对应的全局自由度起始编号
        dof2_start = n2 * ndof_node
        # 填充当前单元的LM列
        LM[0:ndof_node, e] = np.arange(dof1_start, dof1_start + ndof_node)
        LM[ndof_node:, e] = np.arange(dof2_start, dof2_start + ndof_node)
    return LM

def assemble_global_K(model, LM):
    """
    根据LM矩阵，将所有单元刚度矩阵组装为总体刚度矩阵K
    :param model: 模型字典
    :param LM: 对号矩阵
    :return: 总体刚度矩阵 K
    """
    total_dof = model["total_dof"]
    nel = model["nel"]
    K = np.zeros((total_dof, total_dof), dtype=np.float64)

    for e in range(nel):
        # 计算单元刚度矩阵
        Ke = cal_element_stiffness(model, e)
        lm_col = LM[e]  # 当前单元对应的全局自由度编号
        # 直接累加组装：K[lm_a, lm_b] += Ke[a,b]
        for a in range(len(lm_col)):
            for b in range(len(lm_col)):
                K[lm_col[a], lm_col[b]] += Ke[a, b]
    return K

def is_symmetric(mat, tol=1e-8):
    """判断矩阵是否对称"""
    return np.allclose(mat, mat.T, atol=tol)
