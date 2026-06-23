import numpy as np
from element import get_element_geo, cal_stress_force

def post_process(model, d, LM):
    """
    后处理：遍历所有单元，计算并输出单元几何、应力、轴力
    :param model: 模型字典
    :param d: 整体位移向量
    :param LM: 对号矩阵
    """
    nel = model["nel"]
    print("\n" + "="*60)
    print("【后处理结果：各单元几何、应力、轴力】")
    print("="*60)

    for e in range(nel):
        # 1. 提取单元局部位移 de
        lm_col = LM[:, e]
        de = d[lm_col]
        # 2. 单元几何参数
        _, _, _, _, L, c, s = get_element_geo(model, e)
        # 3. 计算应力、轴力
        sigma, N = cal_stress_force(model, e, de, L, c, s)

        # 格式化输出
        print(f"\n>>> 单元 {e+1}：")
        print(f"  单元长度 L = {L:.8f}")
        print(f"  方向余弦 c={c:.8f}, s={s:.8f}")
        print(f"  单元应力 σ = {sigma:.8f}")
        print(f"  单元轴力 N = {N:.8f}")
