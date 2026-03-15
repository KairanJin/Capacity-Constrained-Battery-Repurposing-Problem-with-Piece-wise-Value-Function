# battery_recycling_optimization.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any
import warnings

warnings.filterwarnings('ignore')


# =========================================================
# 带容量约束的 K-means 聚类算法
# =========================================================
class CapacityKMeans:
    """
    带容量约束的 K-means 聚类选址算法

    参数
    ----
    n_clusters : int
        聚类数 K。
    capacity_limits : array-like, shape (K,)
        每个类（站点）的容量上限（允许分配的数据点个数）。
    max_iter_init : int
        阶段 (1) 初始聚类的最大迭代轮数（分配 + 更新中心）。
    max_iter_refine : int
        阶段 (2) 再判定/调整/交换的最大迭代轮数。
    max_exchange_per_cluster : int or None
        尝试交换时，在候选簇内最多检查多少个点。
    random_state : int or None
        随机种子。
    """

    def __init__(
        self,
        n_clusters,
        capacity_limits,
        max_iter_init=20,
        max_iter_refine=10,
        max_exchange_per_cluster=None,
        random_state=None
    ):
        self.n_clusters = int(n_clusters)
        self.capacity_limits = np.asarray(capacity_limits, dtype=int)
        assert self.capacity_limits.shape[0] == self.n_clusters, \
            "capacity_limits 长度必须等于 n_clusters"

        self.max_iter_init = max_iter_init
        self.max_iter_refine = max_iter_refine
        self.max_exchange_per_cluster = max_exchange_per_cluster
        self.random_state = random_state

        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, X):
        """
        在数据 X 上执行带容量约束的 K-means 聚类。
        """
        X = np.asarray(X, dtype=float)
        n_samples, _ = X.shape

        if self.capacity_limits.sum() < n_samples:
            raise ValueError("所有类容量之和小于数据量，无法完成分配。")

        rng = np.random.default_rng(self.random_state)

        # (1) 初始聚类：随机选择 K 个点作为初始中心
        initial_indices = rng.choice(n_samples, size=self.n_clusters, replace=False)
        centers = X[initial_indices].copy()

        labels = -np.ones(n_samples, dtype=int)
        sizes = np.zeros(self.n_clusters, dtype=int)

        # 阶段 (1)：考虑容量限制的初始聚类（分配 + 更新中心）
        for _ in range(self.max_iter_init):
            changed = self._assign_with_capacity(X, centers, labels, sizes)
            centers = self._recompute_centers(X, labels, self.n_clusters, centers)
            if not changed:
                break

        # 阶段 (2)：再判定 + 移动/交换（Ward 准则→总 SSE 变小）
        for _ in range(self.max_iter_refine):
            moved_any = self._refine_with_move_and_exchange(X, labels, centers, sizes)
            if not moved_any:
                break

        self.cluster_centers_ = centers
        self.labels_ = labels
        return self

    # ---------- 内部工具 ----------

    @staticmethod
    def _euclidean_distances(X, centers):
        X_sq = np.sum(X ** 2, axis=1, keepdims=True)          # (n, 1)
        C_sq = np.sum(centers ** 2, axis=1, keepdims=True).T  # (1, k)
        XC = X @ centers.T                                    # (n, k)
        d2 = X_sq + C_sq - 2 * XC
        d2 = np.maximum(d2, 0.0)
        return np.sqrt(d2)

    @staticmethod
    def _recompute_centers(X, labels, n_clusters, old_centers=None):
        n_features = X.shape[1]
        centers = np.zeros((n_clusters, n_features), dtype=float)
        for k in range(n_clusters):
            mask = (labels == k)
            if np.any(mask):
                centers[k] = X[mask].mean(axis=0)
            else:
                if old_centers is not None:
                    centers[k] = old_centers[k]
                else:
                    centers[k] = X[np.random.randint(0, X.shape[0])]
        return centers

    @staticmethod
    def _total_sse(X, labels, centers):
        diff = X - centers[labels]
        return np.sum(diff ** 2)

    def _assign_with_capacity(self, X, centers, labels, sizes):
        n_samples = X.shape[0]
        dist = self._euclidean_distances(X, centers)

        labels.fill(-1)
        sizes[:] = 0

        changed = False
        for i in range(n_samples):
            order = np.argsort(dist[i])  # 最近→最远
            assigned = False
            for k in order:
                if sizes[k] < self.capacity_limits[k]:
                    labels[i] = k
                    sizes[k] += 1
                    changed = True
                    assigned = True
                    break
            if not assigned:
                raise RuntimeError("某个点在考虑容量限制时无法分配，请检查容量设定。")
        return changed

    def _refine_with_move_and_exchange(self, X, labels, centers, sizes):
        n_samples = X.shape[0]
        dist_all = self._euclidean_distances(X, centers)
        old_sse = self._total_sse(X, labels, centers)
        moved_any = False

        for i in range(n_samples):
            r = labels[i]
            d_i = dist_all[i]
            candidate_clusters = np.argsort(d_i)

            for c in candidate_clusters:
                if c == r:
                    # 最近的类就是原类 → 不调整
                    break

                if sizes[c] < self.capacity_limits[c]:
                    # 容量允许 → 尝试直接移动
                    new_labels = labels.copy()
                    new_labels[i] = c
                    new_centers = self._recompute_centers(X, new_labels, self.n_clusters, centers)
                    new_sse = self._total_sse(X, new_labels, new_centers)

                    if new_sse < old_sse:
                        labels[:] = new_labels
                        centers[:] = new_centers
                        sizes[r] -= 1
                        sizes[c] += 1
                        old_sse = new_sse
                        moved_any = True
                    break

                else:
                    # 容量不允许 → 尝试交换
                    indices_c = np.where(labels == c)[0]
                    if len(indices_c) == 0:
                        continue
                    d_to_r_center = np.linalg.norm(X[indices_c] - centers[r], axis=1)
                    order_in_c = np.argsort(d_to_r_center)

                    max_try = self.max_exchange_per_cluster
                    if max_try is None or max_try > len(order_in_c):
                        max_try = len(order_in_c)

                    swapped = False
                    for idx_pos in range(max_try):
                        j = indices_c[order_in_c[idx_pos]]

                        new_labels = labels.copy()
                        new_labels[i] = c
                        new_labels[j] = r
                        new_centers = self._recompute_centers(X, new_labels, self.n_clusters, centers)
                        new_sse = self._total_sse(X, new_labels, new_centers)

                        if new_sse < old_sse:
                            labels[:] = new_labels
                            centers[:] = new_centers
                            old_sse = new_sse
                            moved_any = True
                            swapped = True
                            break

                    if swapped:
                        break

        return moved_any


# =========================================================
# 电池部分
# =========================================================
class BatteryCell:
    """
    电池电芯类，表示单个退役电池单元
    """

    def __init__(self, resistance: float, capacitance: float, quality: float, arrival_stage: int):
        self.resistance = resistance
        self.capacitance = capacitance
        self.quality = quality
        self.arrival_stage = arrival_stage
        self.cell_id = id(self)

    def __repr__(self):
        return f"BatteryCell(R={self.resistance:.3f}, C={self.capacitance:.3f}, Q={self.quality:.3f})"


class BatteryRecyclingOptimizer:
    """
    电池回收优化器类，实现两阶段随机优化过程
    """

    def __init__(self,
                 mu_R: float = 0.05,
                 sigma_R: float = 0.01,
                 mu_C: float = 3.2,
                 sigma_C: float = 0.1,
                 mu_Q: float = 0.8,
                 sigma_Q: float = 0.1,
                 cells_per_stage: int = 100,
                 scrap_value: float = 10.0,
                 beta: float = 1.0,
                 P_d: float = 0.5,
                 cap_first: int = 6,
                 cap_second: int = 4):
        """
        cap_first: 第一步容量约束（例如 6）
        cap_second: 第二步容量约束（例如 4）
        """

        # 分布参数
        self.mu_R = mu_R
        self.sigma_R = sigma_R
        self.mu_C = mu_C
        self.sigma_C = sigma_C
        self.mu_Q = mu_Q
        self.sigma_Q = sigma_Q

        # 系统参数
        self.cells_per_stage = cells_per_stage
        self.scrap_value = scrap_value
        self.beta = beta
        self.P_d = P_d

        # 容量约束参数
        self.cap_first = cap_first
        self.cap_second = cap_second

        # 状态变量
        self.current_stage = 0
        self.available_cells = []
        self.history_data = []

    # ----------------- 基础工具 -----------------

    def generate_cells(self, n_cells: int, stage: int) -> List[BatteryCell]:
        resistances = np.random.normal(self.mu_R, self.sigma_R, n_cells)
        capacitances = np.random.normal(self.mu_C, self.sigma_C, n_cells)
        qualities = np.random.normal(self.mu_Q, self.sigma_Q, n_cells)
        qualities = np.clip(qualities, 0.1, 1.0)

        cells = []
        for i in range(n_cells):
            cell = BatteryCell(
                resistance=resistances[i],
                capacitance=capacitances[i],
                quality=qualities[i],
                arrival_stage=stage
            )
            cells.append(cell)
        return cells

    def cells_to_dataframe(self, cells: List[BatteryCell]) -> pd.DataFrame:
        if not cells:
            return pd.DataFrame(columns=['resistance', 'capacitance', 'quality', 'arrival_stage'])

        data = {
            'resistance': [cell.resistance for cell in cells],
            'capacitance': [cell.capacitance for cell in cells],
            'quality': [cell.quality for cell in cells],
            'arrival_stage': [cell.arrival_stage for cell in cells]
        }
        return pd.DataFrame(data)

    # ----------------- 容量约束分组逻辑 -----------------

    def group_cells_capacity(
        self,
        cells: List[BatteryCell],
        capacity: int,
        random_state: int = 42
    ) -> Tuple[List[List[int]], List[int], np.ndarray]:
        """
        使用容量约束 K-means 对电芯进行聚类：
        - 容量上限 = capacity
        - 满容量（size == capacity）的簇视为“成功分组”
        - 其他簇的电芯视为“剩余电芯”

        返回：
        - groups: List[List[int]]，每个内部 list 是一组的原始索引
        - remaining_indices: List[int]，剩余电芯的原始索引
        - vis_labels: np.ndarray，长度 = len(cells)，
            对于分组成功的电芯，标记为组号 (0,1,2,...)，
            对于剩余电芯，标记为 -1。
        """
        n = len(cells)
        if n == 0:
            return [], [], np.array([], dtype=int)

        # 电芯太少，连一组都凑不出来
        if n < capacity:
            return [], list(range(n)), -np.ones(n, dtype=int)

        # 估计需要的簇数（向上取整），每簇容量为 capacity
        n_clusters = int(np.ceil(n / capacity))
        capacity_limits = [capacity] * n_clusters

        # 特征：用 (R, C) 聚类
        X = np.array([[c.resistance, c.capacitance] for c in cells], dtype=float)

        model = CapacityKMeans(
            n_clusters=n_clusters,
            capacity_limits=capacity_limits,
            max_iter_init=20,
            max_iter_refine=10,
            max_exchange_per_cluster=None,
            random_state=random_state
        )
        model.fit(X)
        labels = model.labels_  # 0 ~ n_clusters-1

        # 根据簇划分
        groups = []
        remaining_indices = []

        for k in range(n_clusters):
            idx_k = np.where(labels == k)[0].tolist()
            if len(idx_k) == capacity:
                groups.append(idx_k)
            elif len(idx_k) > 0:
                remaining_indices.extend(idx_k)

        # 构造可视化标签：组号从 0 开始，未进组的是 -1
        vis_labels = -np.ones(n, dtype=int)
        group_id = 0
        for g in groups:
            for idx in g:
                vis_labels[idx] = group_id
            group_id += 1

        return groups, remaining_indices, vis_labels

    def group_cells_two_stage(
        self,
        cells: List[BatteryCell],
        cap_first: int = None,
        cap_second: int = None
    ) -> Tuple[List[List[int]], List[int], np.ndarray]:
        """
        两阶段分组：
        1）容量约束为 cap_first：
            - 满 cap_first 的簇视为成功分组，先“暂时去掉”
            - 其余为剩余电芯
        2）对剩余电芯再跑一次，容量约束为 cap_second：
            - 满 cap_second 的簇视为成功分组
            - 其余为最终剩余电芯

        如果未传入 cap_first / cap_second，则使用实例属性 self.cap_first / self.cap_second。
        """
        if cap_first is None:
            cap_first = self.cap_first
        if cap_second is None:
            cap_second = self.cap_second

        n = len(cells)
        if n == 0:
            return [], [], np.array([], dtype=int)

        # -------- 第 1 阶段：容量 = cap_first --------
        groups_1, remaining_1, _ = self.group_cells_capacity(
            cells, capacity=cap_first, random_state=42
        )

        # 第 2 阶段在“剩余电芯”子集上操作
        remaining_cells = [cells[i] for i in remaining_1]
        groups_2_rel, remaining_2_rel, _ = self.group_cells_capacity(
            remaining_cells, capacity=cap_second, random_state=43
        )

        # 将第 2 阶段的相对索引映射回原始索引
        groups_2 = [[remaining_1[idx] for idx in g_rel] for g_rel in groups_2_rel]
        final_remaining_indices = [remaining_1[idx] for idx in remaining_2_rel]

        # 合并所有分组
        all_groups = []
        all_groups.extend(groups_1)
        all_groups.extend(groups_2)

        # 构造最终的可视化标签
        vis_labels = -np.ones(n, dtype=int)
        group_id = 0
        for g in all_groups:
            for idx in g:
                vis_labels[idx] = group_id
            group_id += 1

        return all_groups, final_remaining_indices, vis_labels

    # ----------------- 组效用相关 -----------------

    def compute_group_diversity(self, group_data: pd.DataFrame) -> float:
        h_j = len(group_data)
        if h_j < 2:
            return 0.0

        diversity_sum = 0.0
        for i in range(len(group_data)):
            for k in range(len(group_data)):
                if i != k:
                    ri, ci = group_data.iloc[i]['resistance'], group_data.iloc[i]['capacitance']
                    rk, ck = group_data.iloc[k]['resistance'], group_data.iloc[k]['capacitance']
                    diversity_sum += (ri - rk) ** 2 + (ci - ck) ** 2

        return diversity_sum

    def compute_group_utility(self, group_data: pd.DataFrame) -> float:
        h_j = len(group_data)
        if h_j < 2:
            return 0.0
        d_j = self.compute_group_diversity(group_data)
        U_j = self.beta * h_j * self.P_d * d_j
        return U_j

    # ----------------- 蒙特卡洛：等待 vs 报废 -----------------

    def simulate_next_stage_benefit(self, waiting_cells: List[BatteryCell],
                                    n_simulations: int = 20) -> Tuple[float, float, float]:
        scrap_benefit = len(waiting_cells) * self.scrap_value
        simulation_benefits = []

        for _ in range(n_simulations):
            new_cells = self.generate_cells(self.cells_per_stage, self.current_stage + 1)
            combined_cells = waiting_cells + new_cells

            if len(combined_cells) >= 4:
                combined_df = self.cells_to_dataframe(combined_cells)
                # 使用两阶段容量约束 K-means
                groups, remaining_indices, _ = self.group_cells_two_stage(
                    combined_cells,
                    cap_first=self.cap_first,
                    cap_second=self.cap_second
                )

                if groups:
                    group_benefit = 0.0
                    for g_idx in groups:
                        group_data = combined_df.loc[g_idx]
                        group_benefit += self.compute_group_utility(group_data)
                    simulation_benefits.append(group_benefit)
                else:
                    simulation_benefits.append(len(combined_cells) * self.scrap_value)
            else:
                simulation_benefits.append(len(combined_cells) * self.scrap_value)

        expected_benefit = np.mean(simulation_benefits) if simulation_benefits else 0.0
        if simulation_benefits:
            count_better = sum(1 for benefit in simulation_benefits if benefit > scrap_benefit)
            probability_better = count_better / len(simulation_benefits)
        else:
            probability_better = 0.0

        return expected_benefit, probability_better, scrap_benefit

    def make_decision_for_remaining_cells(self, remaining_cells: List[BatteryCell]) -> Dict[str, Any]:
        if not remaining_cells:
            return {
                'decision': 'none',
                'benefit': 0.0,
                'remaining_cells': []
            }

        wait_benefit, probability_better, scrap_benefit = self.simulate_next_stage_benefit(
            remaining_cells, n_simulations=200
        )

        if probability_better >= 0.95:
            decision = 'wait'
            benefit = wait_benefit
            next_stage_cells = remaining_cells
            print(f"  决策: 等待 (95%置信度下等待收益更高)")
            print(f"    等待期望收益: {wait_benefit:.2f}")
            print(f"    报废收益: {scrap_benefit:.2f}")
            print(f"    收益超过报废收益的概率: {probability_better:.2%}")
        else:
            decision = 'scrap'
            benefit = scrap_benefit
            next_stage_cells = []
            print(f"  决策: 报废 (95%置信度下等待收益不显著更高)")
            print(f"    报废收益: {scrap_benefit:.2f}")
            print(f"    等待期望收益: {wait_benefit:.2f}")
            print(f"    收益超过报废收益的概率: {probability_better:.2%}")

        return {
            'decision': decision,
            'benefit': benefit,
            'remaining_cells': next_stage_cells,
            'scrap_benefit': scrap_benefit,
            'wait_benefit': wait_benefit,
            'probability_better': probability_better
        }

    # ----------------- 阶段处理 -----------------

    def process_stage(self, stage_num: int, cells: List[BatteryCell]) -> Dict[str, Any]:
        self.current_stage = stage_num
        print(f"\n{'=' * 50}")
        print(f"阶段 {stage_num} 处理开始")
        print(f"{'=' * 50}")
        print(f"总电芯数量: {len(cells)}")

        if len(cells) < 4:
            print("电芯数量不足，无法形成有效分组。")
            decision_result = self.make_decision_for_remaining_cells(cells)
            return {
                'stage': stage_num,
                'num_groups': 0,
                'groups_info': [],
                'decision': decision_result['decision'],
                'stage_benefit': decision_result['benefit'],
                'remaining_cells': decision_result['remaining_cells']
            }

        # 使用两阶段容量约束 K-means：先 cap_first，再 cap_second
        groups, remaining_indices, vis_labels = self.group_cells_two_stage(
            cells,
            cap_first=self.cap_first,
            cap_second=self.cap_second
        )

        cells_df = self.cells_to_dataframe(cells)
        cells_df['label'] = vis_labels

        # 统计分组信息
        num_groups = len(groups)
        groups_info = []
        total_group_utility = 0.0

        print(f"成功形成 {num_groups} 个电池组:")
        for i, group_indices in enumerate(groups):
            group_data = cells_df.loc[group_indices]
            utility = self.compute_group_utility(group_data)
            total_group_utility += utility

            avg_r = group_data['resistance'].mean()
            avg_c = group_data['capacitance'].mean()
            avg_q = group_data['quality'].mean()

            group_info = {
                'group_id': i + 1,
                'size': len(group_data),
                'utility': utility,
                'avg_resistance': avg_r,
                'avg_capacitance': avg_c,
                'avg_quality': avg_q
            }
            groups_info.append(group_info)

            print(f"  组 {i + 1}: 大小={len(group_data)}, 效用={utility:.4f}, "
                  f"平均R={avg_r:.4f}, 平均C={avg_c:.4f}, 平均Q={avg_q:.4f}")

        print(f"分组总效用: {total_group_utility:.4f}")

        # 处理未分组的电芯
        remaining_cells = [cells[i] for i in remaining_indices] if remaining_indices else []
        print(f"未分组电芯数量: {len(remaining_cells)}")

        decision_result = self.make_decision_for_remaining_cells(remaining_cells)
        stage_benefit = total_group_utility + decision_result['benefit']

        print(f"阶段 {stage_num} 总收益: {stage_benefit:.2f} (分组收益: {total_group_utility:.2f} + "
              f"{decision_result['decision']}收益: {decision_result['benefit']:.2f})")

        self.visualize_clustering(cells_df, vis_labels, stage_num)

        return {
            'stage': stage_num,
            'num_groups': num_groups,
            'groups_info': groups_info,
            'total_group_utility': total_group_utility,
            'decision': decision_result['decision'],
            'decision_benefit': decision_result['benefit'],
            'scrap_benefit': decision_result.get('scrap_benefit', 0),
            'wait_benefit': decision_result.get('wait_benefit', 0),
            'stage_benefit': stage_benefit,
            'remaining_cells': decision_result['remaining_cells']
        }

    def visualize_clustering(self, cells_df: pd.DataFrame, labels: np.ndarray, stage_num: int):
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(cells_df['resistance'], cells_df['capacitance'],
                              c=labels, cmap='viridis', alpha=0.7, s=50)
        plt.xlabel('内阻 (Resistance)')
        plt.ylabel('电容 (Capacitance)')
        plt.title(
            f'阶段 {stage_num} 聚类/分组结果（两阶段容量 K-means：'
            f'{self.cap_first}→{self.cap_second}）'
        )
        plt.colorbar(scatter, label='组标签 (-1 表示未分组)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # ----------------- 多阶段运行 & 总结 -----------------

    def run_optimization(self, num_stages: int = 5) -> List[Dict[str, Any]]:
        print("电池回收优化系统启动")
        print(f"参数设置: 每阶段电芯数={self.cells_per_stage}, 报废收益={self.scrap_value}")
        print(f"分布参数: R~N({self.mu_R}, {self.sigma_R}²), C~N({self.mu_C}, {self.sigma_C}²)")
        print(f"容量约束：第一阶段={self.cap_first}，第二阶段={self.cap_second}")

        results = []
        remaining_cells = []

        for stage in range(1, num_stages + 1):
            new_cells = self.generate_cells(self.cells_per_stage, stage)
            combined_cells = remaining_cells + new_cells

            print(f"\n阶段 {stage}: 上一阶段剩余 {len(remaining_cells)} 个电芯，"
                  f"新到达 {len(new_cells)} 个电芯，合计 {len(combined_cells)} 个电芯")

            stage_result = self.process_stage(stage, combined_cells)
            results.append(stage_result)

            remaining_cells = stage_result['remaining_cells']

            self.history_data.append({
                'stage': stage,
                'total_cells': len(combined_cells),
                'num_groups': stage_result['num_groups'],
                'stage_benefit': stage_result['stage_benefit'],
                'remaining_cells': len(remaining_cells)
            })

        self.print_summary(results)
        return results

    def print_summary(self, results: List[Dict[str, Any]]):
        print(f"\n{'=' * 60}")
        print("优化过程总结")
        print(f"{'=' * 60}")

        total_benefit = sum(result['stage_benefit'] for result in results)
        total_groups = sum(result['num_groups'] for result in results)
        avg_benefit = total_benefit / len(results) if results else 0

        print(f"总阶段数: {len(results)}")
        print(f"总形成组数: {total_groups}")
        print(f"总收益: {total_benefit:.2f}")
        print(f"平均阶段收益: {avg_benefit:.2f}")

        wait_decisions = sum(1 for result in results if result['decision'] == 'wait')
        scrap_decisions = sum(1 for result in results if result['decision'] == 'scrap')
        print(f"等待决策次数: {wait_decisions}")
        print(f"报废决策次数: {scrap_decisions}")


def main():
    """
    主函数，执行多阶段电池回收优化流程
    """

    # ===== 在这里设置容量约束 =====
    cap_first = 6   # 第一步分组容量（例如 6）
    cap_second = 4  # 第二步分组容量（例如 4）

    optimizer = BatteryRecyclingOptimizer(
        mu_R=0.05,
        sigma_R=0.01,
        mu_C=3.2,
        sigma_C=0.1,
        mu_Q=0.8,
        sigma_Q=0.1,
        cells_per_stage=100,
        scrap_value=0.5,
        beta=1.0,
        P_d=0.5,
        cap_first=cap_first,
        cap_second=cap_second
    )

    print(f"当前容量约束设置：第一阶段={cap_first}，第二阶段={cap_second}")

    results = optimizer.run_optimization(num_stages=5)
    print("\n优化完成！")


if __name__ == "__main__":
    main()
