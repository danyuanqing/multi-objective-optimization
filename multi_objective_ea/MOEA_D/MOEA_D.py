# 导入全部依赖（适配pymoo 0.6.0+）
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.problems import get_problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.decomposition.tchebicheff import Tchebicheff
import time

# 1. 定义优化问题（ZDT1，双目标）
problem = get_problem("zdt5")
n_obj = problem.n_obj

# 2. 生成参考方向（核心优化：双目标用12个分区，仅13个方向）
ref_dirs = get_reference_directions("uniform", n_obj, n_partitions=200)  # 从100→12

# 3. 初始化切比雪夫分解类
decomp = Tchebicheff()

# 4. 配置MOEA/D（优化参数，匹配NSGA-II的计算量）
algorithm_moead = MOEAD(
    ref_dirs=ref_dirs,
    decomposition=decomp,
    n_neighbors=10,              # 保持邻居数（10是双目标合理值）
    delta=0.7,                   # 保持delta
    sampling=LatinHypercubeSampling(),
    crossover=SBX(prob=0.8, eta=8),     # 保持交叉参数
    # 核心优化：变异概率改为1/n_var（和NSGA-II一致）
    mutation=PM(eta=12, prob=1/problem.n_var)  # prob≈0.033
)

start = time.time()
res_moead = minimize(
    problem,
    algorithm_moead,
    ('n_gen', 250),
    seed=1,
    verbose=False
)
time_moead = time.time() - start

# 6. 可视化（无需过滤，ZDT1无极端值）
valid_F = res_moead.F  # ZDT1的f2范围是0~1，无需过滤

plot = Scatter(
    title="MOEA/D on ZDT1 (Optimized)",
    figsize=(8, 6),
    dpi=300,
    legend=True
)
plot.add(valid_F, color="red", s=6, label="MOEA/D")
plot.show()
plot.save("moead_zdt1_optimized.png", dpi=300)

# 8. 打印结果
print("="*50)
print("MOEA/D优化成功！耗时大幅降低")
print(f"Pareto前沿点数量：{len(res_moead.F)}")
print(f"决策变量维度：{problem.n_var}")
print(f"目标函数数量：{problem.n_obj}")
print(f"{'运行时间（秒）':<15}         {time_moead:.2f}" )
print("="*50)