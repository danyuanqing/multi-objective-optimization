# 导入全部依赖
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.problems import get_problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.decomposition.tchebicheff import Tchebicheff
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.util.dominator import Dominator

import matplotlib.pyplot as plt
try:
    from pymoo.indicators.igd import IGD
    from pymoo.indicators.hv import HV
except ImportError:
    from pymoo.indicators import IGD
    from pymoo.indicators import Hypervolume as HV
import time
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']  # 支持中文+符号
plt.rcParams['axes.unicode_minus'] = False  # 专门解决减号显示问题
# ===================== 1. 测试配置（ZDT2） =====================
TEST_PROBLEM = "zdt5"
POP_SIZE = 250  # 增大种群规模
N_GEN = 250     # 增加进化代数
SEED = 1
REF_POINT = np.array([1.1, 1.1])

# ===================== 2. 定义问题 =====================
problem = get_problem(TEST_PROBLEM)
n_obj = problem.n_obj
pf = problem.pareto_front()

# ===================== 3. 配置NSGA-II =====================
algorithm_nsga2 = NSGA2(
    pop_size=POP_SIZE,
    sampling=LatinHypercubeSampling(),
    crossover=SBX(prob=1.0, eta=2),
    mutation=PM(eta=5, prob=1/problem.n_var),
    eliminate_duplicates=True
)

# ===================== 4. 配置MOEA/D（核心调整：增大参考方向数） =====================
ref_dirs = get_reference_directions("uniform", n_obj, n_partitions=POP_SIZE)  # 参考方向数从20→30
decomp = Tchebicheff()
algorithm_moead = MOEAD(
    ref_dirs=ref_dirs,
    decomposition=decomp,
    n_neighbors=20,  # 邻居数随种群增大
    delta=0.8,
    sampling=LatinHypercubeSampling(),
    crossover=SBX(prob=1.0, eta=2),
    mutation=PM(eta=5, prob=1 / problem.n_var),
)

# ===================== 5. 运行算法 =====================
# MOEA/D
start = time.time()
res_moead = minimize(problem, algorithm_moead, ('n_gen', N_GEN), seed=SEED, verbose=False)
time_moead = time.time() - start
# NSGA-II
start = time.time()
res_nsga2 = minimize(problem, algorithm_nsga2, ('n_gen', N_GEN), seed=SEED, verbose=False)
time_nsga2 = time.time() - start


# ===================== 6. 计算指标 =====================
# IGD
if pf is not None:
    igd_calc = IGD(pf)
    igd_nsga2 = igd_calc.calc(res_nsga2.F) if hasattr(igd_calc, 'calc') else igd_calc(res_nsga2.F)
    igd_moead = igd_calc.calc(res_moead.F) if hasattr(igd_calc, 'calc') else igd_calc(res_moead.F)
else:
    igd_nsga2 = igd_moead = np.nan

# HV
hv_calc = HV(ref_point=REF_POINT)
hv_nsga2 = hv_calc.calc(res_nsga2.F) if hasattr(hv_calc, 'calc') else hv_calc(res_nsga2.F)
hv_moead = hv_calc.calc(res_moead.F) if hasattr(hv_calc, 'calc') else hv_calc(res_moead.F)

# ===================== 7. 绘图（核心调整：给MOEA/D点加边框） =====================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(figsize=(8, 6), dpi=400)

# 先画NSGA-II（蓝点，无框）
plt.scatter(res_nsga2.F[:, 0], res_nsga2.F[:, 1], color='blue', s=6,
            label=f'NSGA-II (IGD={igd_nsga2:.4f})')
# 后画MOEA/D（红点+黑框，避免被覆盖）
plt.scatter(res_moead.F[:, 0], res_moead.F[:, 1], color='red', s=8, edgecolors='black', linewidths=0.3,
            label=f'MOEA/D (IGD={igd_moead:.4f})')

# 图表属性
plt.title(f'NSGA-II vs MOEA/D on {TEST_PROBLEM}')
plt.xlabel('f1')
plt.ylabel('f2')
plt.ylim(0, 2)
plt.legend(loc='upper right')
plt.tight_layout()

# 显示+保存

plt.savefig(f'nsga2_vs_moead_{TEST_PROBLEM}.png', dpi=400, bbox_inches='tight')
plt.show()

# ===================== 8. 输出结果 =====================
print("="*80)
print(f"多目标算法横向对比结果（{TEST_PROBLEM}）")
print("="*80)
print(f"{'指标':<15} {'NSGA-II':<15} {'MOEA/D':<15} {'优劣判断':<15}")
print("-"*80)
print(f"{'IGD（收敛性）':<15} {igd_nsga2:.4f}       {igd_moead:.4f}       {'NSGA-II更优' if igd_nsga2 < igd_moead else 'MOEA/D更优'}")
print(f"{'HV（分布性）':<15} {hv_nsga2:.4f}       {hv_moead:.4f}       {'NSGA-II更优' if hv_nsga2 > hv_moead else 'MOEA/D更优'}")
print(f"{'运行时间（秒）':<15} {time_nsga2:.2f}         {time_moead:.2f}         {'NSGA-II更快' if time_nsga2 < time_moead else 'MOEA/D更快'}")
print("="*80)