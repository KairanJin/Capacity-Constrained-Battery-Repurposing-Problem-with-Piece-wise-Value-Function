import numpy as np


class CapacityKMeans:
    """
    带容量约束的 K-means 聚类选址算法（对应文中 3.3 节）

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
        在尝试与某一候选簇进行交换时，最多检查该簇中多少个点（按照“距离类 r 最近”的排序依次尝试）。
        如果为 None，则默认为该簇当前点数。
    random_state : int or None
        随机种子，便于复现。

    属性
    ----
    cluster_centers_ : ndarray, shape (K, d)
        最终的类中心坐标。
    labels_ : ndarray, shape (n_samples,)
        每个样本所属的类标签。
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

    # ---------- 公共接口 ----------

    def fit(self, X):
        """
        在数据 X 上执行带容量约束的 K-means 聚类。

        参数
        ----
        X : ndarray, shape (n_samples, n_features)

        返回
        ----
        self
        """
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape

        if self.capacity_limits.sum() < n_samples:
            raise ValueError("所有类容量之和小于数据量，无法完成分配。")

        rng = np.random.default_rng(self.random_state)

        # (1) 初始聚类：随机选择 K 个点作为初始中心
        initial_indices = rng.choice(n_samples, size=self.n_clusters, replace=False)
        centers = X[initial_indices].copy()

        # 容量和 labels 初始化
        labels = -np.ones(n_samples, dtype=int)   # -1 表示尚未分配
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

    # ---------- 内部工具函数 ----------

    @staticmethod
    def _euclidean_distances(X, centers):
        """
        计算所有点到所有中心的欧式距离矩阵。
        返回形状 (n_samples, n_clusters)
        """
        # X: (n, d), centers: (k, d)
        # 使用 (x - c)^2 展开的向量化计算
        X_sq = np.sum(X ** 2, axis=1, keepdims=True)          # (n, 1)
        C_sq = np.sum(centers ** 2, axis=1, keepdims=True).T  # (1, k)
        XC = X @ centers.T                                    # (n, k)
        d2 = X_sq + C_sq - 2 * XC
        d2 = np.maximum(d2, 0.0)
        return np.sqrt(d2)

    @staticmethod
    def _recompute_centers(X, labels, n_clusters, old_centers=None):
        """
        按当前 labels 重新计算每个类的中心（坐标均值）。
        若某类为空，则保持原中心不动（使用 old_centers）。
        """
        n_features = X.shape[1]
        centers = np.zeros((n_clusters, n_features), dtype=float)
        for k in range(n_clusters):
            mask = (labels == k)
            if np.any(mask):
                centers[k] = X[mask].mean(axis=0)
            else:
                # 保留原来的中心，防止出现空类
                if old_centers is not None:
                    centers[k] = old_centers[k]
                else:
                    centers[k] = X[np.random.randint(0, X.shape[0])]
        return centers

    @staticmethod
    def _total_sse(X, labels, centers):
        """
        计算整体的 SSE（sum of squared errors），即 Ward 准则中的“类内离差平方和总和”。
        """
        diff = X - centers[labels]
        return np.sum(diff ** 2)

    # ---------- 阶段 (1)：考虑容量限制的初始聚类 ----------

    def _assign_with_capacity(self, X, centers, labels, sizes):
        """
        在容量限制下，将每个点分配到“最近且有剩余容量”的类中。
        对应文中 (1) 的步骤 ② + ③。
        """
        n_samples = X.shape[0]
        dist = self._euclidean_distances(X, centers)

        # 重置 labels & sizes
        labels.fill(-1)
        sizes[:] = 0

        changed = False
        # 遍历每个点 i
        for i in range(n_samples):
            # 将该点到各中心的距离从小到大排序，依次尝试
            order = np.argsort(dist[i])  # 最近→最远
            assigned = False
            for k in order:
                if sizes[k] < self.capacity_limits[k]:
                    labels[i] = k
                    sizes[k] += 1
                    assigned = True
                    changed = True
                    break
            if not assigned:
                raise RuntimeError("某个点在考虑容量限制时无法分配，请检查容量设定。")

        return changed

    # ---------- 阶段 (2)：再判定 + 移动/交换 ----------

    def _refine_with_move_and_exchange(self, X, labels, centers, sizes):
        """
        对所有点再次判断，进行移动或交换（若目标类容量已满），
        并用“总 SSE 是否减小”作为是否接受交换的准则。
        对应文中 (2) 步骤 ①~⑥。
        """
        n_samples = X.shape[0]
        dist_all = self._euclidean_distances(X, centers)
        old_sse = self._total_sse(X, labels, centers)

        moved_any = False

        for i in range(n_samples):
            r = labels[i]          # 当前所属类 r
            d_i = dist_all[i]      # 点 i 到各中心的距离
            # 将簇按距离从小到大排序（对应“离它最近的类、第二近、第三近……”）
            candidate_clusters = np.argsort(d_i)

            # 逐个考虑候选簇
            moved_or_swapped = False
            for c in candidate_clusters:
                if c == r:
                    # (2)-②-1：最近的类就是原类 → 分配正确 → 不调整 → 跳出考虑下一个点
                    break

                # （最近的候选类 c 的容量是否允许？）
                if sizes[c] < self.capacity_limits[c]:
                    # 直接移动 i 到类 c（文中 (2)-②-2）
                    new_labels = labels.copy()
                    new_labels[i] = c
                    new_centers = self._recompute_centers(X, new_labels, self.n_clusters, centers)
                    new_sse = self._total_sse(X, new_labels, new_centers)
                    # 这里可以直接移动，也可以要求 SSE 更优再移动。
                    # 文中描述是“直接移动”，但你如果想严格遵循 Ward 准则可改成 new_sse < old_sse 才移动。
                    labels[:] = new_labels
                    centers[:] = new_centers
                    sizes[r] -= 1
                    sizes[c] += 1
                    old_sse = new_sse
                    moved_any = True
                    moved_or_swapped = True
                    break

                else:
                    # 容量不允许 → 尝试交换（文中 (2)-②-3）
                    # 先找出类 c 中的点，计算它们到类 r 中心的距离，并排序
                    indices_c = np.where(labels == c)[0]
                    if len(indices_c) == 0:
                        continue

                    # c 簇中的点到 r 类中心的距离
                    d_to_r_center = np.linalg.norm(X[indices_c] - centers[r], axis=1)
                    order_in_c = np.argsort(d_to_r_center)

                    max_try = self.max_exchange_per_cluster
                    if max_try is None or max_try > len(order_in_c):
                        max_try = len(order_in_c)

                    swapped = False
                    for idx_pos in range(max_try):
                        j = indices_c[order_in_c[idx_pos]]

                        # 尝试交换 i ↔ j
                        new_labels = labels.copy()
                        new_labels[i] = c
                        new_labels[j] = r

                        new_centers = self._recompute_centers(X, new_labels, self.n_clusters, centers)
                        new_sse = self._total_sse(X, new_labels, new_centers)

                        # 使用 Ward 准则：若交换后总 SSE 减小，则认为“更优”，接受交换
                        if new_sse < old_sse:
                            labels[:] = new_labels
                            centers[:] = new_centers
                            # sizes 不变（只是两个类各少一个、多一个）
                            old_sse = new_sse
                            moved_any = True
                            moved_or_swapped = True
                            swapped = True
                            break

                    if swapped:
                        break
                    # 若该候选簇 c 中所有尝试都不改进，则再看下一个候选簇

            # 该点 i 的候选簇都尝试完毕（或提前 break），若有移动/交换则转到下一个点
            # 若没有移动/交换，直接进入下一点 i+1
            # （候选簇列表中已经按“最近、第二近、第三近”依次尝试了）

        return moved_any


# ========== 简单示例用法 ==========
if __name__ == "__main__":
    # 生成一些二维数据用于测试
    rng = np.random.default_rng(0)
    X1 = rng.normal(loc=[0, 0], scale=0.5, size=(50, 2))
    X2 = rng.normal(loc=[5, 5], scale=0.5, size=(50, 2))
    X3 = rng.normal(loc=[0, 5], scale=0.5, size=(50, 2))
    X = np.vstack([X1, X2, X3])

    # 三个类，每类容量上限 60（总容量 180 ≥ 150 点）
    model = CapacityKMeans(
        n_clusters=3,
        capacity_limits=[60, 60, 60],
        max_iter_init=10,
        max_iter_refine=10,
        random_state=42
    )
    model.fit(X)

    print("Cluster centers:")
    print(model.cluster_centers_)
    print("Cluster sizes:", np.bincount(model.labels_))
