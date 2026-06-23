# 有限元平衡方程组求解程序 运行说明
## 1. 项目简介
本项目为《2-4 平衡方程组的求解》程序设计作业，复用2.3作业有限元组装模块，自研LDL^T稠密求解器、调用Intel MKL PARDISO稀疏求解器，完成误差、残差、条件数分析与多组验证算例。

## 2. 环境与依赖
- Python: 3.9 ~ 3.12
- 依赖包：numpy, scipy, pypardiso, mkl, matplotlib
- 安装命令：pip install -r requirements.txt

## 3. 目录说明
- src/: 全部源代码（fea_23为2.3复用模块）
- cases/: JSON格式算例输入文件
- results/: 运行结果、日志、绘图
- report/: 作业报告PDF

## 4. 运行步骤
1. 进入 src/ 目录：cd src
2. 运行主程序：python main.py
3. 结果自动输出至 ../results/ 目录。

## 5. 功能说明
1. LDL^T稠密求解：对称正定矩阵分解、主元检测、前代/回代；
2. 误差分析：残差、相对误差、条件数、病态矩阵测试；
3. 稀疏求解：COO/CSR转换、Intel MKL PARDISO调用；
4. 有限元衔接：完整对接2.3桁架模型，输出位移、应力、反力；
5. Poisson方程有限元：Q4单元手动组装、稀疏求解、误差可视化。

## 6. 注意事项
1. 所有数组下标从 0 开始；
2. 核心LDL^T算法完全自研，未调用第三方求解库；
3. PARDISO依赖Intel MKL，必须执行 pip install mkl；
4. 大规模Poisson算例建议在8G及以上内存设备运行。