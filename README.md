# Capacity-Constrained Battery Repurposing Problem with Piece-wise Value Function

## 项目概述

本项目研究和求解**容量约束下的电池重组问题**，该问题涉及将退役电动汽车电池单元重新组合成新的电池组。电池单元由容量 (C) 和内阻 (R) 两个特征描述。

项目采用**两阶段优化框架**：
- **内层问题**：给定一批电池单元，如何分组打包以最大化收益
- **外层问题**：在动态到达环境下，决定何时以及如何报废低质量库存电池

---

## 问题背景

### 电池单元特征
每个电池单元由两个属性描述：
- **C**：容量 (Capacity)，通常服从正态分布 N(200, 20²)
- **R**：内阻 (Resistance)，通常服从正态分布 N(50, 5²)

### 分组规则
- 每个电池组必须包含 **K 个电池单元**（默认 K=8）
- 电池组必须满足**容量约束**：组内方差 $\delta \leq \bar{\delta}$
- 分组数量上限为 $k_{max}$

### 分段价值函数
根据电池组的质量指标 $\phi = \boldsymbol{w}^T \boldsymbol{\mu}$，分为三个等级：
- **P1 等级**：$\phi \geq \theta_1$，价值 $P_1=10.0$
- **P2 等级**：$\theta_2 \leq \phi < \theta_1$，价值 $P_2=6.0$
- **P3 等级**：$\theta_3 \leq \phi < \theta_2$，价值 $P_3=3.0$
- **无效**：$\phi < \theta_3$，价值 $0$

### 目标函数
最大化总收益：
$$ \max \sum_{g \in G} [V(\phi_g) - \lambda \delta_g] $$

其中：
- $V(\phi_g)$ 是分段价值函数
- $\lambda$ 是方差惩罚系数
- $\delta_g$ 是组内方差

---

## 项目结构

```
two-stage/
├── config.py                  # 配置文件
├── data_generator.py          # 数据生成器
├── utils.py                   # 工具函数
├── main_two_stage.py          # 两阶段系统主程序
├── run_two_stage_experiment.py # 实验运行脚本
├── upper_bound.py             # 上界计算
├── multi_inner_opt.py         # 多内层优化
│
├── heuristics/                # 内层启发式算法
│   ├── rrp_kmeans.py          # K-means 聚类算法
│   ├── rrp_kmeans_vns.py      # K-means + 变邻域搜索(VNS)
│   ├── rrp_grasp.py           # GRASP 算法
│   ├── rrp_ga.py              # 遗传算法(GA)
│   ├── rrp_column_generation.py # 列生成算法
│   ├── solve_rrp_lns.py       # 大邻域搜索(LNS)
│   ├── neural_rl.py           # 神经强化学习
│   └── hybrid_rl_ga.py        # 混合RL+GA算法
│
├── outer/                     # 外层优化模块
│   ├── tsrah.py               # 两阶段鲁棒启发式(TSRH)
│   ├── arrival.py             # 到达过程生成器
│   └── test_outer.py          # 外层测试
│
├── results/                   # 结果存储目录
├── Old_code/                  # 旧代码存档
└── requirements.txt           # 依赖包列表
```

---

## 算法说明

### 内层求解算法

| 算法 | 描述 | 适用场景 |
|------|------|----------|
| **K-means** | 基于K-means的聚类分组 | 快速近似解 |
| **VNS** | K-means + 变邻域搜索 | 中等规模问题 |
| **GRASP** | 贪婪随机自适应搜索 | 多样性好的解 |
| **GA** | 遗传算法 | 全局搜索 |
| **Column Generation** | 列生成精确算法 | 中小规模精确求解 |
| **LNS** | 大邻域搜索 | 大规模问题 |

### 外层决策算法

#### TSRH (Two-Stage Robust Heuristic)
TSRH 是用于动态报废决策的两阶段鲁棒启发式算法：

1. **阈值评估**：对每个候选阈值 $\eta$ 进行随机模拟（Rollout）
2. **分层筛选**：通过多层筛选，逐步淘汰低效阈值
3. **最终选择**：选择具有最高平均收益的阈值

**参数说明**：
- `E_thresholds`：候选阈值列表
- `H_scrap`：报废决策周期
- `m_list`：每层的模拟次数，如 [2, 4, 8]
- `rho`：保留比例（0<rho<1）

#### 阶段性上界
通过假设未来信息完全已知（Clairvoyant），计算可达到的理论上界，用于评估算法性能。

---

## 安装与运行

### 环境要求
- Python 3.9+
- NumPy, Pandas, Matplotlib
- SciPy（可选，用于某些算法）

### 安装依赖

```bash
pip install numpy pandas matplotlib scipy
```

或使用虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### 快速开始

#### 1. 运行两阶段系统示例

```bash
python main_two_stage.py
```

这将在 20 个时期内模拟两阶段系统，比较 KMEANS、VNS、GRASP、GA 四种内层算法的性能。

#### 2. 运行实验脚本

```bash
python run_two_stage_experiment.py
```

#### 3. 计算上界

```bash
python upper_bound.py
```

---

## 配置说明

### 主要配置参数 (`config.py`)

```python
@dataclass
class ExperimentConfig:
    run_kmeans: bool = True           # 是否运行K-means
    run_kmeans_vns: bool = True        # 是否运行VNS
    run_grasp: bool = False            # 是否运行GRASP
    run_ga: bool = True                # 是否运行GA
    run_lns: bool = True               # 是否运行LNS
    base_seed: int = 42                # 随机种子

@dataclass
class ProblemConfig:
    n_cells: int = 500                 # 初始电池单元数量
    K: int = 8                         # 每组电池单元数
    k_max: int = 30                    # 最大分组数
    delta_bar: float = 0.1             # 容量约束阈值
    w: Tuple[float, float] = (0.5, 0.5) # 质量权重
    lambda_penalty: float = 0.05       # 方差惩罚系数
    theta1: float = 0.5                # P1等级阈值
    theta2: float = 0.0                # P2等级阈值
    theta3: float = -0.5               # P3等级阈值
    P1: float = 10.0                   # P1等级价值
    P2: float = 6.0                    # P2等级价值
    P3: float = 3.0                    # P3等级价值

@dataclass
class DataConfig:
    mu_C: float = 200.0               # 容量均值
    sigma_C: float = 20.0             # 容量标准差
    mu_R: float = 50.0                # 内阻均值
    sigma_R: float = 5.0              # 内阻标准差
```

---

## 输出结果

### 两阶段系统运行输出示例

```
[ONLINE-VNS] t= 1 | arrivals=130 | packs=10 | I_t^+= 40 | scrap=  0 | I_t++= 40 | R_grp=   42.50 | R_scr=   0.00 | eta=nan
[ONLINE-VNS] t= 2 | arrivals=130 | packs=10 | I_t^+= 40 | scrap=  0 | I_t++= 40 | R_grp=   45.30 | R_scr=   0.00 | eta=nan
[ONLINE-VNS] t= 3 | arrivals=130 | packs=10 | I_t^+= 40 | scrap=  0 | I_t++= 40 | R_grp=   38.20 | R_scr=   0.00 | eta=nan
[ONLINE-VNS] t= 4 | arrivals=130 | packs=10 | I_t^+= 40 | scrap=  0 | I_t++= 40 | R_grp=   51.10 | R_scr=   0.00 | eta=nan
[ONLINE-VNS] t= 5 | arrivals=130 | packs=10 | I_t^+= 40 | scrap=  5 | I_t++= 35 | R_grp=   44.80 | R_scr=  25.00 | eta=115.00
...
```

### 算法对比表格

| Method | Online Group Reward | Online Scrap Reward | Online Total Reward | Gap to UB (%) |
|--------|--------------------|-------------------|-------------------|--------------|
| KMEANS | 852.30             | 125.00            | 977.30            | 12.5%        |
| VNS    | 920.45             | 115.00            | 1035.45           | 5.8%         |
| GRASP  | 895.20             | 120.00            | 1015.20           | 8.2%         |
| GA     | 935.60             | 110.00            | 1045.60           | 3.5%         |

### 结果文件

运行结果将保存在 `results/` 目录下，包括：
- 每个时期的详细统计数据
- 各算法的性能对比图表
- 最终解的分组信息

---

## 数学模型

### 内层问题（RRP）

**变量**：
- $x_{ij} \in \{0,1\}$：电池单元 $i$ 是否分配到组 $j$
- $y_j \in \{0,1\}$：组 $j$ 是否被使用

**模型**：
$$
\begin{align}
\max \quad & \sum_{j=1}^{k_{max}} \left[ V(\phi_j) - \lambda \delta_j \right] y_j \\
\text{s.t.} \quad & \sum_{j=1}^{k_{max}} x_{ij} \leq 1, \quad \forall i \\
& \sum_{i=1}^{n} x_{ij} = K y_j, \quad \forall j \\
& \delta_j = \frac{1}{K} \sum_{i=1}^{n} x_{ij} \| \boldsymbol{z}_i - \boldsymbol{\mu}_j \|^2, \quad \forall j \\
& \delta_j \leq \bar{\delta}, \quad \forall j \\
& \phi_j = \boldsymbol{w}^T \boldsymbol{\mu}_j, \quad \forall j \\
& x_{ij}, y_j \in \{0,1\}
\end{align}
$$

### 外层问题

在时期 $t$，已知当前库存 $I_t^+$，需要决定报废决策 $D_t$：

$$
\max_{\eta \in E} \left[ s_0 |D_t| + \sum_{\tau=1}^{H_{scrap}} \gamma^\tau \mathbb{E}[R_{t+\tau}] \right]
$$

其中：
- $\eta$ 是报废阈值
- $s_0$ 是单位报废收益
- $\gamma$ 是折扣因子
- $R_{t+\tau}$ 是未来 $\tau$ 时期的内层收益

---

## 算法细节

### K-means + VNS

1. **初始化**：使用 K-means 获得初始分组
2. **邻域操作**：
   - 移动：将一个单元移动到另一组
   - 交换：交换两组中的单元
3. **变邻域**：依次使用不同大小的邻域进行搜索
4. **停止条件**：最大迭代次数或连续无改进次数

### GRASP

1. **贪婪构建**：每次从受限候选列表(RCL)中选择最佳或次优单元
2. **局部搜索**：对构建的解进行局部优化
3. **重复**：执行多次迭代，选择最佳解

### 遗传算法 (GA)

1. **编码**：将分组方案编码为染色体
2. **选择**：锦标赛选择
3. **交叉**：两点交叉
4. **变异**：随机交换
5. **精英保留**：保留最优个体

---

## 常见问题

### Q1: 如何修改电池单元数量？
修改 `config.py` 中的 `ProblemConfig.n_cells` 参数。

### Q2: 如何更改报废周期？
修改 `main_two_stage.py` 中的 `H_scrap` 参数。

### Q3: 如何添加新的内层算法？
在 `heuristics/` 目录下创建新文件，实现相同的接口，然后在 `solve_inner_rrp()` 函数中添加对应的分支。

### Q4: 如何可视化结果？
运行 `main_two_stage.py` 会自动生成对比图表，或者使用 `matplotlib` 自定义绘图。

---

## 参考文献

本项目基于以下研究：
- 两阶段鲁棒启发式(TSRH)框架
- 容量约束下的聚类问题
- 动态库存管理优化

---

## 联系方式

如有问题或建议，欢迎提交 Issue 或 Pull Request。

---

## 许可证

本项目仅供学术研究使用。
