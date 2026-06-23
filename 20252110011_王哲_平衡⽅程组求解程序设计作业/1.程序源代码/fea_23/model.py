import json
import numpy as np

def read_model(json_path):
    """
    前处理：读取JSON输入文件，解析有限元模型所有参数
    :param json_path: 输入JSON文件路径
    :return: 模型字典model
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 基础维度参数
    model = {}
    model["title"] = data["Title"]
    model["nsd"] = data["nsd"]         # 空间维度：1一维，2二维
    model["ndof_per_node"] = data["ndof"]  # 单节点自由度
    model["nnp"] = data["nnp"]         # 总节点数
    model["nel"] = data["nel"]         # 总单元数
    model["nen"] = data["nen"]         # 单单元节点数(桁架恒为2)
    model["total_dof"] = model["nnp"] * model["ndof_per_node"]  # 整体总自由度

    # 材料与截面参数
    model["E"] = np.array(data["E"], dtype=np.float64)       # 弹性模量
    model["A"] = np.array(data["CArea"], dtype=np.float64)    # 截面积

    # 节点坐标
    model["x"] = np.array(data["x"], dtype=np.float64)
    model["y"] = np.array(data["y"], dtype=np.float64) if "y" in data else np.zeros_like(model["x"])

    # 单元连接 IEN: 外部1-based → 内部0-based
    IEN_1base = np.array(data["IEN"], dtype=int)
    model["IEN"] = IEN_1base - 1  # 节点编号转0索引

    # 位移边界条件：固定自由度、固定位移 (1-based → 0-based)
    fix_dof_1 = np.array(data["fixed_dof"], dtype=int)
    model["fixed_dof"] = fix_dof_1 - 1
    model["fixed_val"] = np.array(data["fixed_value"], dtype=np.float64)

    # 节点载荷：受力自由度、载荷值 (1-based → 0-based)
    force_dof_1 = np.array(data["force_dof"], dtype=int)
    model["force_dof"] = force_dof_1 - 1
    model["force_val"] = np.array(data["force_value"], dtype=np.float64)

    # 初始化整体载荷向量
    model["F"] = np.zeros(model["total_dof"], dtype=np.float64)
    for dof, val in zip(model["force_dof"], model["force_val"]):
        model["F"][dof] = val

    return model
