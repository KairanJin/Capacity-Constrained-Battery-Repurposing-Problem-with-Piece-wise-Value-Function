# 数值实验设计：验证 RRP_KMEANS_VNS 算法优良性

## 1. 实验目标

通过系统的数值实验，验证所提 **RRP_KMEANS_VNS** 算法（K-Means + 变邻域搜索）相对于其他启发式算法在解质量、运行时间和稳定性方面的优良性。实验分为两个层次：

- **实验一（Inner RRP 基准实验）**：在固定电芯集合上比较各算法的纯优化能力
- **实验二（Two-Stage 完整系统实验）**：在动态到达 + 废弃决策环境中比较各算法的集成效果

---

## 2. 实验一：Inner RRP 基准实验

### 2.1 对比算法

| 编号 | 算法 | 角色定位 | 预期表现 |
|------|------|---------|---------|
| 1 | RRP_KMEANS | 简单基线 | 速度最快，质量最低 |
| 2 | **RRP_KMEANS_VNS** | **所提算法** | **质量与速度的最佳平衡** |
| 3 | RRP_GRASP | 最强竞争者 | 高质量，中等速度 |
| 4 | RRP_GA | 进化算法代表 | 高质量，慢速 |
| 5 | RRP_SA | 模拟退火代表 | 高质量，中慢速 |
| 6 | RRP_MS_KMEANS_VNS | 多起点变体 | 与所提算法最直接竞争 |
| 7 | RRP_COMBINE_REPAIR | 组合修复算法 | 中等质量，快速 |
| 8 | RRP_COLUMN_GENERATION | 列生成（理论保证） | 中高质量，速度可变 |
| 9 | RRP_GUROBI_ENUM | 精确解（小规模） | 最优解，验证启发式间隙 |

### 2.2 实例集设计

**问题参数固定：** K=8, delta_bar=0.8, w=(0.5, 0.5), lambda_penalty=0.05, theta=(0.5, 0.0, -0.5), P=(10.0, 6.0, 3.0)

**电芯分布：** mu_C=200, sigma_C=20, mu_R=50, sigma_R=5

| 档位 | N (电芯数) | 种子数 | GUROBI 可用 | 目的 |
|------|-----------|--------|------------|------|
| 小 | 30, 40, 50 | 20 | 是 | 验证最优性间隔 |
| 中 | 100, 200, 300 | 20 | 否 | 典型应用场景 |
| 大 | 500 | 10 | 否 | 可扩展性测试 |

### 2.3 假设检验

| 假设 | 内容 | 检验方法 |
|------|------|---------|
| H1 | VNS 的平均 reward 显著高于所有其他启发式算法 | 配对 t 检验 (alpha=0.05) |
| H2 | VNS 与最优解的间隙在可接受范围内 (<5%) | 最优性间隔分析 |
| H3 | VNS 的运行时间显著低于 GA/SA 等元启发式 | 配对 t 检验 (runtime) |
| H4 | VNS 的解质量稳定性 (std) 不低于其他算法 | 方差比较 + F 检验 |
| H5 | 所有算法整体存在显著差异 | ANOVA F 检验 |

### 2.4 统计检验方法

- **配对 t 检验**：同实例同种子配对，VNS vs 每个竞争者
- **Cohen's d 效应量**：量化差异幅度（d>0.8 为大效应）
- **ANOVA + Tukey HSD**：多算法整体差异显著性及事后成对比较
- **最优性间隔**：gap_pct = 100 * (optimal - heuristic) / optimal

---

## 3. 实验二：Two-Stage 完整系统实验

### 3.1 实验设计

在动态到达 + TSRH 废弃决策的完整两阶段系统中，比较各算法作为内层求解器的表现。

| 参数 | 值 |
|------|-----|
| n_periods | 20 |
| arrivals_per_period | 130 |
| H_scrap | [4, 5, 6] |
| seeds | 10 个 (42-51) |
| 内层方法 | 8 种启发式 (不含 GUROBI) |

### 3.2 采集指标

| 指标 | 说明 |
|------|------|
| online_total_reward | 在线策略总收益 |
| ub_total_reward | 分阶段 clairvoyant 上界 |
| gap_pct | 与上界的间隙百分比 |
| online_total_scrap | 废弃电芯数量 |
| online_avg_packs | 平均 pack 数 |
| total_runtime | 总运行时间 |

---

## 4. 实施与运行

### 4.1 文件结构

```
two-stage/
├── experiment_utils.py          # 统计检验 + Excel 生成
├── experiment_inner_rrp.py      # 实验一主脚本
├── experiment_two_stage.py      # 实验二主脚本
├── research.md                  # 本文件（实验设计文档）
└── results/
    ├── experiment_inner_rrp.xlsx
    └── experiment_two_stage.xlsx
```

### 4.2 运行命令

```bash
# 激活虚拟环境
.venv\Scripts\activate

# 实验一：快速验证 (N=30,40, 3 种子)
python experiment_inner_rrp.py --quick

# 实验一：完整运行
python experiment_inner_rrp.py

# 实验一：指定规模
python experiment_inner_rrp.py --sizes 100 200 300 --seeds 20

# 实验二：快速验证
python experiment_two_stage.py --quick

# 实验二：完整运行
python experiment_two_stage.py

# 跳过特定算法
python experiment_inner_rrp.py --skip GUROBI_ENUM COLUMN_GENERATION
```

### 4.3 输出内容

Excel 报告包含以下 sheet：

| Sheet | 内容 |
|-------|------|
| README | 实验参数说明 |
| Raw_Results | 每实例每算法原始数据 |
| Summary | 汇总统计（均值/标准差/最优/最差/中位数） |
| Statistical_Tests | t 检验、ANOVA、Tukey HSD、效应量 |
| Optimality_Gap | 小规模实例与 GUROBI 对比 |
| ChartData_* | 图表数据（箱线图、可扩展性曲线） |
| Charts | 嵌入式图表 |

---

## 5. 预期结果与论文贡献

### 5.1 预期排名（按 reward）

1. GUROBI_ENUM（最优解，仅小规模）
2. VNS / GRASP / GA / SA（高质量启发式，彼此接近）
3. MS_KMEANS_VNS / COMBINE_REPAIR（中等偏上）
4. COLUMN_GENERATION（理论保证但实践中可能稍弱）
5. KMEANS（基线，明显落后）

### 5.2 所提算法的预期优势

- **vs KMEANS**：证明 VNS 五类邻域搜索的增量价值
- **vs GRASP**：证明系统化变邻域搜索优于贪婪随机构造
- **vs GA/SA**：证明单解精细化搜索在运行时间上的优势（质量相当但更快）
- **vs MS_KMEANS_VNS**：证明单起点深度搜索优于多起点浅搜索
- **vs GUROBI**：证明启发式在大规模实例上的实用性和接近最优的质量

### 5.3 论文中的图表建议

1. **箱线图**：各算法 reward 分布（按 N 分组）
2. **散点图**：reward vs runtime 权衡（质量-效率前沿）
3. **折线图**：最优性间隔 vs N（小规模）
4. **表格**：汇总统计 + 统计显著性标记（*/**/***）
5. **柱状图**：Two-Stage gap_pct 比较
