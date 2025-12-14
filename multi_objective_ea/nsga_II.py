from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.lhs import LatinHypercubeSampling  # 和MOEA/D对齐采样
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
import time
import numpy as np

# 1. 统一配置（所有参数和MOEA/D对齐）
TEST_PROBLEM = "zdt5"
POP_SIZE = 300  # 核心：种群规模100
N_GEN = 250
SEED = 1

# 2. 定义问题
problem = get_problem(TEST_PROBLEM)
n_obj = problem.n_obj

# 3. 配置NSGA-II（参数和MOEA/D完全对齐）
algorithm_nsga2 = NSGA2(
    pop_size=POP_SIZE,
    sampling=LatinHypercubeSampling(),  # 和MOEA/D一致（原代码是FloatRandomSampling，改为LHS对齐）
    crossover=SBX(prob=0.8, eta=8),     # 和MOEA/D交叉参数一致
    mutation=PM(eta=12, prob=1/problem.n_var),  # 和MOEA/D变异参数一致
    eliminate_duplicates=True
)

# 4. 运行并计时
start = time.time()
res_nsga2 = minimize(
    problem,
    algorithm_nsga2,
    ('n_gen', N_GEN),
    seed=SEED,
    verbose=False
)
time_nsga2 = time.time() - start

# 5. 可视化
plot_nsga2 = Scatter(
    title="NSGA-II on ZDT1 (Fair Compare)",
    figsize=(8, 6),
    dpi=300
)
plot_nsga2.add(res_nsga2.F, color="blue", s=6, label="NSGA-II")
plot_nsga2.show()
plot_nsga2.save("nsga2_zdt1_fair.png", dpi=300)

# 6. 打印结果
print("="*60)
print("NSGA-II 结果（pop_size=100）")
print("="*60)
print(f"Pareto前沿点数量：{len(res_nsga2.F)}")
print(f"决策变量维度：{problem.n_var}")
print(f"目标函数数量：{problem.n_obj}")
print(f"运行时间（秒）：{time_nsga2:.2f}")
print("="*60)