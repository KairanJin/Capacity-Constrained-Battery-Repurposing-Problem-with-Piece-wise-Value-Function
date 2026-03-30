import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


class BatteryCell:
    """
    电芯类，表示单个电池单元
    """

    def __init__(self, capacitance, resistance, stage):
        self.capacitance = capacitance
        self.resistance = resistance
        self.stage = stage  # 所属阶段

    def __repr__(self):
        return f"BatteryCell(C={self.capacitance:.2f}, R={self.resistance:.2f}, Stage={self.stage})"


class BatteryGroup:
    """
    电池组类，表示聚类后的一个电池组
    """

    def __init__(self, cells, group_id):
        self.cells = cells
        self.group_id = group_id
        self.centroid_capacitance = np.mean([cell.capacitance for cell in cells])
        self.centroid_resistance = np.mean([cell.resistance for cell in cells])

    def calculate_diversity(self):
        """
        计算组内差异度指标

        dj = Σ(i,k∈Gj) [(ci - ck)² + (ri - rk)²]
        """
        diversity = 0
        n = len(self.cells)
        for i in range(n):
            for k in range(n):
                if i != k:
                    ci, ri = self.cells[i].capacitance, self.cells[i].resistance
                    ck, rk = self.cells[k].capacitance, self.cells[k].resistance
                    diversity += (ci - ck) ** 2 + (ri - rk) ** 2
        return diversity

    def __repr__(self):
        return f"BatteryGroup(ID={self.group_id}, Cells={len(self.cells)}, Diversity={self.calculate_diversity():.2f})"


class BatteryOptimizationSystem:
    """
    电池优化系统类，实现两阶段随机优化过程
    """

    def __init__(self, mu_C, sigma_C, mu_R, sigma_R, num_cells_per_stage=50):
        """
        初始化系统参数

        参数:
        mu_C: 电容均值
        sigma_C: 电容标准差
        mu_R: 内阻均值
        sigma_R: 内阻标准差
        num_cells_per_stage: 每阶段新到达电芯数量
        """
        self.mu_C = mu_C
        self.sigma_C = sigma_C
        self.mu_R = mu_R
        self.sigma_R = sigma_R
        self.num_cells_per_stage = num_cells_per_stage
        self.available_cells = []  # 当前可用的电芯
        self.history_groups = []  # 历史所有分组
        self.stage = 0  # 当前阶段

    def generate_cells(self, num_cells):
        """
        生成指定数量的电芯

        参数:
        num_cells: 需要生成的电芯数量

        返回:
        list of BatteryCell: 新生成的电芯列表
        """
        capacitances = np.random.normal(self.mu_C, self.sigma_C, num_cells)
        resistances = np.random.normal(self.mu_R, self.sigma_R, num_cells)

        cells = []
        for i in range(num_cells):
            cell = BatteryCell(capacitances[i], resistances[i], self.stage)
            cells.append(cell)

        return cells

    def stage_one_clustering(self, n_clusters=5):
        """
        第一阶段：电芯聚类与组建

        参数:
        n_clusters: 聚类数量

        返回:
        list of BatteryGroup: 聚类后的电池组列表
        """
        # 合并新到达的电芯和上阶段剩余的电芯
        new_cells = self.generate_cells(self.num_cells_per_stage)
        self.available_cells.extend(new_cells)

        print(f"阶段 {self.stage}: 总共 {len(self.available_cells)} 个可用电芯")

        if len(self.available_cells) < n_clusters:
            print("可用电芯数量少于聚类数，跳过聚类")
            return []

        # 提取特征数据用于K-Means聚类
        features = np.array([[cell.capacitance, cell.resistance] for cell in self.available_cells])

        # 使用K-Means进行聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(features)

        # 根据聚类结果创建电池组
        groups = []
        for cluster_id in range(n_clusters):
            cluster_cells = [self.available_cells[i] for i in range(len(self.available_cells))
                             if labels[i] == cluster_id]
            if cluster_cells:  # 只有当组不为空时才创建
                group = BatteryGroup(cluster_cells, cluster_id)
                groups.append(group)

        self.history_groups.extend(groups)
        return groups

    def stage_two_decision(self, groups, selection_criteria='diversity'):
        """
        第二阶段：基于差异度指标的决策

        参数:
        groups: 电池组列表
        selection_criteria: 选择标准 ('diversity' 或其他)

        返回:
        list of BatteryGroup: 被选中的电池组
        """
        if not groups:
            return []

        # 根据差异度指标排序
        if selection_criteria == 'diversity':
            # 选择差异度最低的组（最一致的组）
            selected_group = min(groups, key=lambda g: g.calculate_diversity())
        else:
            # 默认选择第一个组
            selected_group = groups[0]

        print(
            f"阶段 {self.stage}: 选择电池组 {selected_group.group_id}，差异度: {selected_group.calculate_diversity():.2f}")

        # 从可用电芯中移除被选中的电芯
        selected_cell_ids = set(id(cell) for cell in selected_group.cells)
        self.available_cells = [cell for cell in self.available_cells
                                if id(cell) not in selected_cell_ids]

        return [selected_group]

    def run_optimization(self, num_stages=5, n_clusters=5):
        """
        运行完整的两阶段优化过程

        参数:
        num_stages: 总阶段数
        n_clusters: 每阶段聚类数
        """
        selected_groups_history = []

        for stage in range(num_stages):
            self.stage = stage

            # 第一阶段：电芯聚类与组建
            groups = self.stage_one_clustering(n_clusters)

            # 显示各组信息
            if groups:
                print(f"阶段 {stage} 聚类结果:")
                for group in groups:
                    print(
                        f"  组 {group.group_id}: {len(group.cells)} 个电芯, 差异度 = {group.calculate_diversity():.2f}")

            # 第二阶段：决策
            selected_groups = self.stage_two_decision(groups)
            selected_groups_history.extend(selected_groups)

            print(f"阶段 {stage} 结束，剩余 {len(self.available_cells)} 个电芯\n")

        return selected_groups_history


# 示例使用
if __name__ == "__main__":
    # 设置系统参数
    mu_C = 3.2  # 电容均值 (mAh)
    sigma_C = 0.1  # 电容标准差
    mu_R = 0.05  # 内阻均值 (ohm)
    sigma_R = 0.01  # 内阻标准差

    # 创建优化系统
    system = BatteryOptimizationSystem(mu_C, sigma_C, mu_R, sigma_R, num_cells_per_stage=30)

    # 运行优化过程
    selected_groups = system.run_optimization(num_stages=5, n_clusters=4)

    # 输出最终结果统计
    print("=" * 50)
    print("优化过程完成!")
    print(f"总共处理了 {len(system.history_groups)} 个候选电池组")
    print(f"选择了 {len(selected_groups)} 个电池组用于实际应用")

    # 可视化结果
    if system.history_groups:
        diversities = [group.calculate_diversity() for group in system.history_groups]
        plt.figure(figsize=(10, 6))
        plt.hist(diversities, bins=20, alpha=0.7, color='blue')
        plt.xlabel('组内差异度')
        plt.ylabel('频次')
        plt.title('所有候选电池组的差异度分布')
        plt.grid(True, alpha=0.3)
        plt.show()
