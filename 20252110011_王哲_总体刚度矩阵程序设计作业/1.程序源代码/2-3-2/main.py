import sys
import json
import numpy as np

# ------------------------------
# 模型读取函数（适配你的JSON结构）
# ------------------------------
def read_model(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        model = json.load(f)
    # 节点编号从1转为0-based
    model['IEN'] = np.array(model['IEN']) - 1
    # 二维桁架总自由度 = 节点数 × 每个节点自由度
    model['total_dof'] = model['nnp'] * model['ndof']
    return model

# ------------------------------
# 二维桁架单元刚度矩阵函数
# ------------------------------
def cal_element_stiffness(model, e):
    E = model['E'][e]
    A = model['CArea'][e]
    IEN = model['IEN']
    x = model['x']
    y = model['y']
    n1, n2 = IEN[e, 0], IEN[e, 1]
    x1, y1 = x[n1], y[n1]
    x2, y2 = x[n2], y[n2]
    dx = x2 - x1
    dy = y2 - y1
    L = np.hypot(dx, dy)
    if L <= 1e-10:
        raise ValueError(f"单元{e}长度为0，模型错误！")
    c = dx / L
    s = dy / L
    ke = E * A / L
    Ke = ke * np.array([
        [c*c,  c*s, -c*c, -c*s],
        [c*s,  s*s, -c*s, -s*s],
        [-c*c, -c*s, c*c,  c*s],
        [-c*s, -s*s, c*s,  s*s]
    ], dtype=np.float64)
    return Ke

# ------------------------------
# 组装矩阵函数（适配二维）
# ------------------------------
def gen_LM(model):
    nel = model['nel']
    ndof_node = model['ndof']  # 二维桁架每个节点2个自由度
    IEN = model['IEN']
    elem_dof = 4  # 2节点 × 2自由度 = 4
    LM = np.zeros((elem_dof, nel), dtype=int)
    for e in range(nel):
        n1, n2 = IEN[e, 0], IEN[e, 1]
        dof1_start = n1 * ndof_node
        dof2_start = n2 * ndof_node
        LM[0:2, e] = np.arange(dof1_start, dof1_start + 2)
        LM[2:4, e] = np.arange(dof2_start, dof2_start + 2)
    return LM

def assemble_global_K(model, LM):
    total_dof = model['total_dof']
    nel = model['nel']
    K = np.zeros((total_dof, total_dof), dtype=np.float64)
    for e in range(nel):
        Ke = cal_element_stiffness(model, e)
        lm_col = LM[:, e]
        for a in range(len(lm_col)):
            for b in range(len(lm_col)):
                K[lm_col[a], lm_col[b]] += Ke[a, b]
    return K

def is_symmetric(mat, tol=1e-8):
    return np.allclose(mat, mat.T, atol=tol)

def judge_singular(mat, tol=1e-8):
    if mat.shape[0] != mat.shape[1]:
        return True
    try:
        det = np.linalg.det(mat)
    except:
        return True
    return abs(det) < tol

# ------------------------------
# 求解器函数（适配你的边界条件）
# ------------------------------
def solve_reduction(model, K):
    fixed_dof = np.array(model['fixed_dof']) - 1  # 转为0-based
    force_dof = np.array(model['force_dof']) - 1
    force_value = np.array(model['force_value'])
    total_dof = model['total_dof']
    free_dof = np.setdiff1d(np.arange(total_dof), fixed_dof)
    
    # 组装载荷向量
    F = np.zeros(total_dof)
    F[force_dof] = force_value
    
    # 缩减刚度矩阵和载荷向量
    K_ff = K[np.ix_(free_dof, free_dof)]
    F_f = F[free_dof]
    
    # 求解位移
    d_f = np.linalg.solve(K_ff, F_f)
    d = np.zeros(total_dof)
    d[free_dof] = d_f
    
    # 计算约束反力
    R = np.zeros(len(fixed_dof))
    for i, dof in enumerate(fixed_dof):
        R[i] = np.dot(K[dof, :], d) - F[dof]
    
    return d, R, K_ff

# ------------------------------
# 后处理函数（二维桁架轴力计算）
# ------------------------------
def post_process(model, d, LM):
    nel = model['nel']
    IEN = model['IEN']
    E = model['E']
    A = model['CArea']
    x = model['x']
    y = model['y']
    
    print("\n===== 单元应力与轴力 =====")
    for e in range(nel):
        n1, n2 = IEN[e, 0], IEN[e, 1]
        x1, y1 = x[n1], y[n1]
        x2, y2 = x[n2], y[n2]
        dx = x2 - x1
        dy = y2 - y1
        L = np.hypot(dx, dy)
        c = dx / L
        s = dy / L
        
        lm_col = LM[:, e]
        u = d[lm_col]
        # 轴力 N = EA/L * ((u2x - u1x)*c + (u2y - u1y)*s)
        N = (E[e] * A[e] / L) * ((u[2] - u[0])*c + (u[3] - u[1])*s)
        sigma = N / A[e]
        
        print(f"单元{e+1} 应力: {sigma:.4f}, 轴力: {N:.4f}")

# ------------------------------
# 主函数
# ------------------------------
def main(json_path):
    # ========= 1. 前处理：读取有限元模型 =========
    model = read_model(json_path)
    print("===== 1. 前处理：读取有限元模型 =====")
    print(f"模型名称：{model['Title']}")
    print(f"总节点数：{model['nnp']}，总单元数：{model['nel']}")
    print(f"总自由度：{model['total_dof']}")

    # ========= 2. 生成LM对号矩阵 =========
    print("\n===== 2. 生成LM对号矩阵 =====")
    LM = gen_LM(model)
    print("LM 对号矩阵：")
    print(LM)

    # ========= 3. 组装总体刚度矩阵 K =========
    print("\n===== 3. 组装总体刚度矩阵 K =====")
    K = assemble_global_K(model, LM)
    print("总体刚度矩阵 K：")
    print(K.round(4))

    # 检查对称性
    sym_flag = is_symmetric(K)
    print(f"总体刚度矩阵是否对称：{sym_flag}")

    # 检查奇异
    sing_flag = judge_singular(K)
    print(f"施加边界条件前，总体刚度矩阵是否奇异：{sing_flag}")

    # ========= 4. 求解位移与约束反力 =========
    print("\n===== 4. 求解位移与约束反力 =====")
    d, R, K_FF = solve_reduction(model, K)
    print("节点位移向量 d (全局自由度，0-based):")
    for idx, val in enumerate(d):
        print(f"自由度 {idx+1:2d} (内部 {idx:2d}): {val:>8.6f}")
    
    print("\n约束自由度对应的约束反力 R:")
    fixed_dof = np.array(model['fixed_dof']) - 1
    for i, dof in enumerate(fixed_dof):
        print(f"固定自由度 {dof+1:2d}: 约束反力 = {R[i]:>8.6f}")

    # 检查缩减后矩阵是否奇异
    sing_FF = judge_singular(K_FF)
    print(f"\n缩减后刚度矩阵K_FF是否奇异：{sing_FF}")

    # ========= 5. 后处理：单元应力、轴力 =========
    post_process(model, d, LM)
    print("\n程序运行结束！")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用方式：python main.py 模型文件.json")
        print("示例：python main.py case2_2d_truss.json")
        sys.exit(1)
    json_file = sys.argv[1]
    main(json_file)
