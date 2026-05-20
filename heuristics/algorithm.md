# 启发式算法基本思想

本文档描述了 `heuristics` 文件夹中每种启发式算法的核心思想。所有算法均针对**电芯重组问题（RRP, Battery Cell Reorganization Problem）**设计，目标是将 N 个电芯划分为最多 k_t 个 pack，每个 pack 包含恰好 K 个电芯，在满足组内方差约束（delta <= delta_bar）的前提下，最大化所有 pack 的总收益（reward）。

---

## 1. RRP_KMEANS — K-Means 聚类 + 收益改进交换

**基本思想：** 将电芯重组视为带容量约束的聚类问题，先利用 SSE 风格的 K-Means 快速形成组内一致性高的分组，再通过组间 swap 提升总收益，最后对不完整组进行重组以最大化满组数量。

### 1.1 收益函数（核心数学模型）

每个 pack（组）的收益由以下公式计算：

1. **组内方差 delta：** `delta = mean(||X[i] - mu||^2)` for i in group，其中 `mu = mean(X[group])` 是组的质心。delta 衡量组内电芯的一致性。
2. **质量指标 phi：** `phi = dot(w, mu)`，即质心在权重向量 w 上的投影值，代表组的整体质量。
3. **阶梯定价 value：** 根据 phi 落入的区间确定基础收益：
   - `phi >= theta1` → `value = P1`（最高档）
   - `phi >= theta2` → `value = P2`
   - `phi >= theta3` → `value = P3`
   - `phi < theta3` → `value = 0`
4. **组收益 reward：** `reward = value - lambda_penalty * delta`，即阶梯定价减去组内方差的惩罚项。

**可行性约束：** 一个 pack 必须同时满足 `len(group) == K` 且 `delta <= delta_bar` 才是可行的。

### 1.2 Stage 1：SSE 风格初始化（L1 次迭代）

1. **随机初始化中心：** 从 n 个电芯中随机选择 k_t 个作为初始聚类中心。
2. **分配阶段（`_assign_to_nearest_nonfull`）：** 遍历每个电芯，计算其到 k_t 个中心的欧氏距离平方，按距离从近到远依次尝试分配到第一个未满的组（容量上限 K）。若所有组都满了，该电芯归入 leftover。
3. **更新阶段（`_update_centers`）：** 对每个非空组，将其中心更新为组内电芯的均值（质心）。
4. **收敛判断：** 计算所有中心最大位移 `shift = max(||new_centers[j] - centers[j]||)`，若 `shift < tol` 则提前终止；否则继续下一轮，最多 L1 轮。

此阶段等价于带容量约束的标准 K-Means（SSE 最小化），目标是让组内电芯在特征空间中尽可能紧凑。

### 1.3 Stage 2：收益改进交换（L2 次迭代）

在 Stage 1 形成的满 K 组之间执行局部搜索，以提升总收益：

1. **遍历离群电芯：** 对每个属于满组的电芯 z，计算其到当前组质心的距离 `dist_r`。
2. **候选组筛选：** 找出所有质心距离比 `dist_r` 更近的满组，按距离升序排列。
3. **枚举 1-1 交换：** 对每个候选组 r2，遍历其所有电芯 zh，尝试交换 (z, zh)：
   - 构造新组 `new_g1 = (g_r \ {z}) ∪ {zh}` 和 `new_g2 = (g_r2 \ {zh}) ∪ {z}`
   - 检查两组交换后是否仍然可行（size == K 且 delta <= delta_bar）
   - 计算交换后的收益增量 `gain = (new_reward_r + new_reward_r2) - (old_reward_r + old_reward_r2)`
   - 若 `gain > 1e-12`（严格正增益），记录最佳交换对。
4. **first-improvement 接受：** 一旦发现任何正增益交换，立即应用并返回 `True`，重新从第一个电芯开始扫描。
5. **终止条件：** 遍历完所有电芯均无改进，或达到 L2 轮上限。

此阶段类似于局部搜索中的 pair-swap 邻域，但以收益（而非距离）为优化目标，同时保持方差约束。

### 1.4 不完整组重组

Stage 2 结束后，部分组可能不满 K 个电芯。重组策略：

1. **分离：** 将满 K 的组保留不动，将不满 K 的组中的所有电芯收集到一个池中。
2. **子聚类：** 若池中电芯数 >= K，则计算可形成的新组数 `n_new = pool_size // K`，在池内重新运行一次 K-Means（随机选择 n_new 个初始中心，执行最近邻分配）。
3. **合并：** 将新生成的组与之前保留的满组合并。若仍有剩余电芯，作为单独的 leftover 组附加。
4. **迭代：** 重复此过程直到不完整组总数 < K 或组数不再变化。

**顺序调整说明：** 将重组放在 Stage 2 之后，是因为先对已有满组进行收益优化，再处理不完整组，避免重组产生的新组被 Stage 2 重新打乱。

### 1.5 可行性过滤与输出

最终对所有组进行统一筛选：

- 仅保留 `len(group) == K` 且 `delta <= delta_bar` 的组
- 计算总收益 `sum(rewards)`、pack 数量、平均 delta 和平均 phi
- 未被任何可行 pack 使用的电芯归入 leftover

**核心优势：** 计算速度快，利用聚类的几何特性快速找到组内一致性高的分组，适合做快速基线。

---

## 2. RRP_KMEANS_VNS — K-Means + 变邻域搜索（VNS）

**基本思想：** 在 K-Means 初始解的基础上，系统地探索多种邻域结构进行局部搜索，且**仅针对 P2/P3 级别的 pack 进行优化，P1 组全程不参与搜索**（避免打乱已达最高质量等级的 pack）。

- **初始解：** 与 RRP_KMEANS 相同的 K-Means 聚类过程。
- **P1 组锁定机制：** 候选组筛选（`_select_candidate_group_ids`）和合作组筛选（`_nearest_partner_group_ids`）均排除 `phi >= theta1` 的 P1 组。所有五类邻域操作的操作对象仅限于 P2/P3 级别的 pack。
- **优先级排序：** 在 P2/P3 组中，根据 pack 的收益和距下一价格阈值的距离，对 pack 进行优先级评分，优先搜索"最值得调整"的 pack。
- **五类邻域操作（N1-N5）：**
  - **N1（1-1 交换）：** 在两个 P2/P3 pack 之间交换单个电芯，采用 first-improvement 策略（找到第一个改进即接受）。优先选择离群电芯（距质心远的电芯）进行交换。
  - **N2（leftover 交换）：** 用 leftover 中的电芯替换 P2/P3 pack 中的离群电芯，优先选择靠近质心且质量贡献高的 leftover 电芯。
  - **N3（2-2 交换）：** 在两个 P2/P3 pack 之间同时交换两个电芯，探索更大的邻域空间。
  - **N4（破坏-修复）：** 选择收益最低的若干个 P2/P3 pack 拆解，将电芯放回候选池，然后以贪婪方式重新构建新 pack。
  - **N5（合并-分裂，默认关闭）：** 将两个 P2/P3 pack 合并后重新枚举所有可能的均衡划分（当 K 较小时可行，K=8 时计算量过大）。
- **两阶段搜索：** Phase A 反复执行 N1/N2/N3 直至无法改进；Phase B 在停滞时触发 N4 大邻域搜索；Phase C 可选触发 N5。

**核心优势：** 多邻域结构避免了单一邻域容易陷入的局部最优，P1 锁定机制避免了对已达最高质量等级 pack 的无效扰动，聚焦搜索资源于有提升空间的 P2/P3 组。

---

## 3. RRP_GA — 遗传算法（Genetic Algorithm）

**基本思想：** 模拟自然进化过程，通过种群的交叉、变异和选择迭代优化。

### 3.1 核心数据结构

- **个体表示：** 每个个体是一个解 `{groups, leftover, fitness}`，其中 `groups` 是可行 pack 列表（每个 pack 为电芯索引列表），`leftover` 是未分配电芯，`fitness` 为可行 pack 的总收益。
- **组缓存（reward_cache）：** 使用 `tuple(sorted(group))` 作为键，缓存每个 pack 的 `{feasible, reward, phi, delta}`，避免在不同个体评估中重复计算 delta/phi。
- **去重机制：** 使用 `_solution_key`（将所有 pack 排序后构成的元组）在初始化和演化中避免重复个体。

### 3.2 种群初始化（混合策略）

初始种群按优先级分三阶段构建：

1. **K-Means-VNS 高质量种子：** 运行 `solve_rrp_kmeans_vns`（缩减参数：L1=12, max_vns_iter=8, max_no_improve=3）生成 `n_vns_seeds` 个种子解（默认 `max(2, min(3, population_size // 4))`），每个种子使用不同随机种子。
2. **普通 K-Means 多样性种子：** 运行一次 `solve_rrp_kmeans`（L1=12, L2=6）生成一个几何聚类导向的解。
3. **随机贪婪填充：** 重复调用 `_random_greedy_individual`，将电芯池随机打乱后用 `_greedy_repair` 逐步构造 pack，直到种群满。若去重后种群仍不足，允许重复个体作为兜底。

### 3.3 贪婪构造与修复（`_greedy_repair`）

在初始化、交叉和变异中均使用的 pack 构建子程序：

1. **种子选择：** 从可用电芯中选择质量最高（`X @ w` 最大）的电芯作为 pack 种子。
2. **候选池构建：** 计算其余电芯到种子的距离和质量评分（`-distance + 0.05 * quality`），取前 `neighbor_limit` 个作为候选池。
3. **组合搜索（`_sample_combinations`）：** 若候选池的 `C(n, K-1)` 组合数 <= `sample_combination_limit`，枚举所有组合；否则随机采样 `sample_combination_limit` 个唯一组合。
4. **最优 pack 选择：** 遍历所有候选组合，选择收益最高的 pack 加入解中。
5. **迭代：** 重复直到电芯不足 K 个或 pack 数达到 `k_t` 上限。

### 3.4 锦标赛选择

从种群中随机选取 `tournament_size` 个不同个体，选择 `fitness` 最高者作为父代。

### 3.5 收益偏置交叉（Reward-Biased Crossover, `_reward_biased_crossover`）

1. **Pack 收集与去重：** 收集两个父代的所有可行 pack，按 pack 的 sorted 索引去重（若同一 pack 在两个父代中出现，取收益更高的那个）。
2. **收益排序与分层：** 按 pack 收益降序排序，前 `inherit_top_ratio`（默认 60%）为"高收益层"，其余为"低收益层"。
3. **分层遗传：**
   - 高收益层 pack：85% 概率被遗传给子代（若电芯无冲突）
   - 低收益层 pack：40% 概率被遗传给子代
4. **贪婪修复：** 将未覆盖的电芯用 `_greedy_repair` 补充为新 pack，完成子代构造。

### 3.6 针对性破坏-修复变异（Targeted Destroy-Repair, `_targeted_destroy_repair_mutation`）

1. **Pack 评分：** 对每个 pack 计算 `score = reward - 0.05 * gap`，其中 `gap` 是距下一价格阈值的距离（P1 组 gap=999）。
2. **候选选择：** 按 score 升序排列，选取最低的几个 pack 作为候选。70% 概率从最低 score 的候选中选择，30% 概率随机选择。
3. **破坏：** 将选中的 pack 拆解，电芯放回候选池（与 leftover 合并）。
4. **修复：** 用 `_greedy_repair` 从候选池重新构建 pack。

### 3.7 选择性局部搜索（Selective Local Search）

以 `local_search_prob` 概率对当代最优子代执行：

1. **1-1 交换（`_swap_local_search_once`）：** 对收益最低的 `group_candidate_limit` 个 pack，各选距质心最远的 `cell_candidate_limit` 个离群电芯，与其余 pack 的离群电芯尝试交换，first-improvement 策略（找到第一个正增益交换即接受并返回）。
2. **Leftover 替换（`_leftover_replace_local_search_once`）：** 若交换无改进，尝试用 leftover 中质量最高的 `leftover_candidate_limit` 个电芯替换 pack 中的离群电芯，同样 first-improvement。

### 3.8 演化主循环

1. **精英保留：** 每代将 `elitism_size` 个最优个体直接复制到下一代。
2. **子代生成：** 重复锦标赛选择 + 交叉/变异，直到种群满。
3. **局部搜索注入：** 对最优子代以 `local_search_prob` 概率执行局部搜索。
4. **停滞检测：** 若连续 `stall_limit` 代最优适应度改进小于 `min_improve`，提前终止。

### 3.9 后处理：残余打包（Residual Packing）

GA 主循环结束后，对最优个体执行 `residual_pack_repair`，从 leftover 中反复提取正收益的可行 pack，最大化 leftover 利用。

**核心优势：** 种群全局搜索能力强，收益偏置交叉能有效组合不同解中的优质 pack，贪婪修复确保每个子代都是可行解。

---

## 4. RRP_SA — 自适应模拟退火（Adaptive Simulated Annealing）

**基本思想：** 结合模拟退火的概率接受机制与 VND 强化搜索，允许以一定概率接受劣解以逃离局部最优。

### 4.1 核心数据结构

- **解表示：** 当前解由 `groups`（可行 pack 列表）+ `leftover`（未分配电芯）+ `current_reward`（总收益）表示。
- **GroupStats 增量统计：** 每个 pack 维护一个 `GroupStats` 对象，记录 `sum_vec`（电芯特征和）、`sum_sq`（电芯范数平方和）、`w_dot_sum`（权重方向投影和），使 swap 操作的 delta/phi 计算复杂度从 O(K·d) 降为 O(d)。
- **Tabu 列表：** 使用 `tabu_set` + `tabu_list`（FIFO 队列）管理禁忌对，禁忌键为 `(min(cell_a, cell_b), max(cell_a, cell_b))`，有效期 `tabu_tenure` 次迭代。

### 4.2 多起点初始化

1. **K-Means 起点：** 用 `n_init_starts` 个不同随机种子运行 `_kmeans_solution`（L1=15, tol=1e-4），每次随机选择 `max(k_t, (n+K-1)//K)` 个初始中心，执行带容量约束的最近邻分配，最多 L1 轮迭代。
2. **贪婪起点：** 运行 `_greedy_solution`，每轮选择质量最高（`X @ w` 最大）的电芯作为种子，在候选中选择 `-distance + 0.05 * quality` 评分最高的 K-1 个电芯组成 pack。
3. **可行性过滤 + 择优：** 对所有候选解过滤掉 `delta > delta_bar` 的 pack，选择总收益最高者作为 SA 初始解。

### 4.3 自适应温度估计（`_estimate_initial_temperature`）

通过采样随机移动计算劣化量分布，使初始温度与问题实例规模自适应匹配：

1. **采样：** 随机生成 `n_samples=50` 次 1-1 swap 和 `n_samples/2` 次 leftover swap。
2. **收集劣化值：** 仅收集 `dE < 0` 的移动的 `|dE|`。
3. **温度计算：** `T0 = mean(|dE|) / ln(2)`，使得 `exp(-mean_de / T0) ≈ 0.5`，即平均接受概率约 50%。
4. **兜底：** 若无劣化移动，返回 `T0 = 1.0`。

### 4.4 增量评估（O(d) 复杂度）

**Swap 增量计算（`_swap_rewards`）：** 对两个 pack a 和 b 交换电芯 `cell_a <-> cell_b`：
```
new_sum_sq_a = stats_a.sum_sq - ||X[cell_a]||^2 + ||X[cell_b]||^2
new_sum_vec_a = stats_a.sum_vec - X[cell_a] + X[cell_b]
new_delta_a = new_sum_sq_a / K - ||new_sum_vec_a||^2 / K^2
new_phi_a = (stats_a.w_dot_sum - w·X[cell_a] + w·X[cell_b]) / K
```
若 `new_delta_a > delta_bar` 则判定不可行，否则计算 `reward = piecewise_value(phi) - lambda * delta`。

**Leftover 增量计算（`_leftover_swap_reward`）：** 同理，仅更新单个 pack 的统计量。

### 4.5 随机邻域移动生成器

每轮 SA 迭代按概率随机选择一种邻域类型生成候选：

1. **1-1 交换（50% 概率）：** 随机选两个不同 pack 各一个电芯，检查 tabu 状态和可行性，计算 `dE`。最多尝试 20 次，找到第一个可行非禁忌移动即返回。
2. **Leftover 交换（30% 概率）：** 随机选一个 pack 的一个电芯和一个 leftover 电芯，检查可行性。最多尝试 20 次。
3. **2-2 交换（20% 概率）：** 随机选两个不同 pack 各两个电芯，互相交换，完整评估（非增量）。最多尝试 15 次。

若所有尝试均未找到可行移动，跳过该轮迭代。

### 4.6 SA 接受准则

- **改进移动（dE >= 0）：** 直接接受。
- **劣化移动（dE < 0）：** 以概率 `exp(dE / T)` 接受（`T > 1e-10` 时）。
- **Tabu 记录：** 仅 1-1 swap 接受后将电芯对加入 tabu 列表，有效期 `tabu_tenure=25` 次迭代。

### 4.7 周期性 VND 强化

每隔 `vnd_interval=200` 次迭代，执行系统的最佳改进局部搜索（`_vnd_intensification`）：

1. **N1 穷举 1-1 交换：** 遍历所有 pack 对 (a, b) 的所有电芯对，使用增量公式计算 `dE`，记录最大正增益移动（`dE > 1e-12`），应用后重新进入 N1。
2. **N2 穷举 leftover 交换：** 若 N1 无改进，遍历所有 pack 电芯与所有 leftover 电芯的组合，记录最大正增益移动，应用后重新进入 N1。
3. **终止：** 若 N1 和 N2 均无改进，退出 VND。最多执行 `max_vnd_rounds=5` 轮。

### 4.8 再加热机制（Reheating）

当连续 `reheating_stall=500` 次迭代未改进最优解时：
- 将温度提升为 `T = T0 * reheating_ratio`（默认 `T0 * 3.0`）
- 重置 `stall_count = 0`
- 最多执行 `max_reheats=3` 次

### 4.9 几何冷却

每轮迭代末尾执行 `T *= cooling_rate`（默认 0.995），下限 `min_temperature=1e-4`。

### 4.10 后处理：残余打包

SA 主循环结束后，对最优解执行 `residual_pack_repair`（`max_rounds=20, seed_candidate_limit=12, neighbor_candidate_limit=20`），从 leftover 中提取额外收益。

**核心优势：** 概率接受机制能有效跳出局部最优，自适应温度和再加热使算法在不同阶段保持合适的探索强度，GroupStats 增量评估使 swap 操作高效（O(d)），VND 周期性强化保证局部搜索精度。

---

## 5. RRP_GRASP — 贪婪随机自适应搜索（GRASP）

**基本思想：** 通过多轮"贪婪随机构造 + 局部搜索"的迭代过程，每轮记录历史最优解。

- **构造阶段（GRASP 贪婪随机构造）：**
  - 从高 RCL（候选列表）中随机选择种子电芯
  - 逐步向组中添加电芯：计算每个候选电芯的边际收益（partial_score 增量），选择边际收益最高的前 RCL_size 个构成候选列表，从中随机选择一个加入
  - 使用增量统计（GroupStats）实现 O(d) 复杂度的 delta/phi 计算
  - 重复构造直至无法形成更多可行 pack
- **局部搜索阶段：** 采用 first-improvement 策略：
  - 1-1 交换：在收益最低的 pack 与其余 pack 之间交换离群电芯
  - leftover 替换：用 leftover 中高质量电芯替换 pack 中的离群电芯
- **多起点：** 重复构造 + 局部搜索 n_starts 次，保留全局最优解。
- **残余打包：** 构造后先执行 residual_pack_repair，再执行局部搜索。

**核心优势：** 随机构造提供了良好的解多样性，增量统计使评估效率极高（swap 操作为 O(1)）。

---

## 6. RRP_COLUMN_GENERATION — 列生成算法（Column Generation）

**基本思想：** 将问题分解为主问题（选择最优 pack 组合）和子问题（发现高价值 pack），通过迭代生成"列"（候选 pack）。

- **初始列集：** 随机选择 n_starts 个种子电芯，在其局部邻域（最近 neighbor_size 个电芯）中枚举所有 K 组合，筛选出可行的 pack 作为初始列。
- **受限主问题（RMP）线性松弛：** 使用 PuLP + CBC 求解 LP 松弛，获得每个电芯的对偶价格 pi 和 pack 数量约束的对偶价格 sigma。
- **定价子问题（寻找正 reduced cost 列）：**
  - reduced_cost = reward(g) - sum(pi_i for i in g) - sigma
  - **局部枚举法：** 在种子电芯的邻域中搜索 reduced cost > 0 的新 pack
  - **Gurobi 法（可选）：** 当 Gurobi 可用时，使用混合整数规划在候选池中精确求解子问题，以线性化近似方式处理非线性 reward 函数
- **迭代：** 当无法找到正 reduced cost 的新列时终止。
- **最终 IP 求解：** 将变量从连续改为整数（Binary），求解 0-1 整数规划得到最终解。

**核心优势：** 列生成框架理论上能保证 LP 松弛的最优性，适合大规模组合优化问题。

---

## 7. RRP_COMBINE_REPAIR — 组合修复算法

**基本思想：** 通过多源种子解生成 pack 池，再从 pack 池中重新组合构建更优解。

- **种子解生成：** 使用 K-Means-VNS、普通 K-Means 和随机贪婪方法生成多个种子解。
- **精英保留：** 评估所有种子解，保留 fitness 最高的 elite_keep 个作为精英解集。
- **Pack 池提取：** 从所有精英解中提取所有可行 pack，按收益排序构成 pack 池。
- **重组：** 从 pack 池中选择 pack 构建新解：
  - 高收益 pack（top 70%）以 85% 概率继承
  - 低收益 pack 以 35% 概率继承
  - 未覆盖电芯通过贪婪修复填补
- **轻量修复：** 对重组后的解执行一次 1-1 交换和一次 leftover 替换。
- **多轮迭代：** 重复重组 + 修复 n_recombine_rounds 次，每轮更新精英解集。
- **最终残余打包：** 可选执行 residual_pack_repair 进一步优化。

**核心优势：** 不依赖传统交叉/变异算子，直接从高质量解中提取和重组优质 pack，搜索效率高。

---

## 8. RRP_MS_KMEANS_VNS — 多起点 K-Means + VNS

**基本思想：** 通过多样化 K-Means 初始解 + 对每个初始解执行聚焦 VNS 搜索，提高找到全局最优的概率。

- **多样化初始解：**
  - 一半种子使用纯 K-Means（不同随机种子）
  - 另一半使用随机扰动 K-Means（shuffle_frac=30% 的电芯随机重新分配），增加解的多样性
- **聚焦 VNS：** 对每个候选解执行 VNS 搜索（N1/N2/N3 轻邻域 + N4 大邻域 + N5 合并分裂），但参数较 `rrp_kmeans_vns` 更紧凑以控制总时间。
- **N5 改进版：** 不再完全枚举所有 C(2K, K) 种划分，而是使用质心距离启发式分割和随机扰动，大幅降低 K=8 时的计算量。
- **后处理：** 对最优解执行残余打包 + 额外 N4 破坏-修复轮次。

**核心优势：** 多起点策略克服了单一 K-Means 初始解质量不稳定的问题，VNS 精细搜索进一步提升解的质量。

---

## 9. RRP_GUROBI_EXACT — Gurobi 精确求解器

**基本思想：** 使用 Gurobi MIP 求解器直接构建数学规划模型，求得全局最优解或带最优性保证的近似解。

- **方法选择（method=auto）：**
  - **枚举法（enumeration）：** 当 C(n,K) 可管理时，枚举所有满足 delta 约束的可行 pack，然后求解集合划分 IP（set-partitioning）。保证全局最优。适用于 N <= 50, K <= 6。
  - **直接 MIP 法（mip）：** 直接构建混合整数二次约束规划模型：
    - 二元变量 x[i,k] 表示电芯 i 分配到组 k
    - 连续变量 mu[k,d] 表示组 k 的质心
    - 二元变量 tier[k,j] 表示组 k 的价格等级
    - 二次约束：`sum x[i,k]*||X[i]||^2 - K*||mu[k]||^2 <= K*delta_bar`（方差约束的展开形式）
    - Big-M 线性化：piecewise 阶梯函数的等级选择
    - Gurobi 通过 NonConvex=2 处理非凸二次约束，全局求解
    - 适用于 N <= 20, K <= 3（受免费许可证规模限制）
- **求解器配置：** 支持 time_limit、mip_gap、threads 等参数，可在最优性和运行时间之间权衡。

**核心优势：** 提供理论最优解和最优性间隔（MIP Gap），是评估启发式算法质量的基准。免费 Gurobi 许可证对 NonConvex 问题有规模限制，大规模实例请使用枚举法或购买完整许可证。

---

## 10. RESIDUAL_PACKING — 残余打包修复（辅助模块）

**基本思想：** 从 leftover 电芯中反复提取正收益的可行 pack，榨取剩余价值。

- **贪婪构造 + 多起点：** 选择质量最高的若干电芯作为种子，对每个种子在其邻域中以贪婪方式逐步构建 pack，尝试多种随机排序以增加多样性。
- **增量统计优化：** 使用 GroupStats 实现 O(d) 复杂度的 delta/phi 增量计算。
- **多轮迭代：** 每轮从 leftover 中提取一个最优 pack，直至无法再形成正收益的可行 pack 或达到 pack 数量上限 k_t。

**定位：** 作为所有主要算法的后处理模块，用于充分利用 leftover 电芯。

---

## 算法对比总结

| 算法 | 类型 | 搜索策略 | 优势 | 适用场景 |
|------|------|----------|------|----------|
| RRP_KMEANS | 聚类 | 几何聚类 + 局部交换 | 速度最快 | 快速基线 |
| RRP_KMEANS_VNS | 元启发式 | 多邻域变邻域搜索 | 局部搜索能力强 | 中等规模 |
| RRP_GA | 进化算法 | 种群交叉变异 | 全局搜索能力强 | 大规模 |
| RRP_SA | 元启发式 | 概率接受 + 退火 | 跳出局部最优能力强 | 复杂地形 |
| RRP_GRASP | 构造+改进 | 贪婪随机 + 局部搜索 | 构造质量好 | 多起点并行 |
| RRP_COLUMN_GENERATION | 精确/启发式 | 列生成 + 线性规划 | 理论保证 | 高质量要求 |
| RRP_COMBINE_REPAIR | 组合优化 | pack 池重组 | 组合现有优质解 | 快速改进 |
| RRP_MS_KMEANS_VNS | 元启发式 | 多起点 + VNS | 初始解多样性 + 精细搜索 | 平衡速度和质量 |
| RRP_GUROBI_EXACT | 精确求解 | MIP / 枚举 + 集合划分 | 全局最优 + 最优性证明 | 小规模基准 / 验证 |
