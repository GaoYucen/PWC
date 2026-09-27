from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .metrics import DEFAULT_OVERLAP_THRESHOLD, route_accuracy_from_overlap

_NO_PREDECESSOR = -9999


@dataclass
class RouteBatchResult:
    accuracy: float
    failures: int
    per_order_accuracy: list[float]
    per_order_overlap: list[float]


class RoadRouter:
    """Shortest-path engine with paper-faithful physical-length overlap."""

    def __init__(
        self,
        city_dir: str | Path,
        overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD,
    ):
        city_dir = Path(city_dir)
        z = np.load(city_dir / "topology.npz", mmap_mode="r")
        self.link_ids = z["link_ids"]
        self.node_ids = z["node_ids"]
        self.edge_u = z["edge_u"]
        self.edge_v = z["edge_v"]
        self.sort_order = z["sort_order"]
        self.group_starts = z["group_starts"]
        self.indices = z["indices"]
        self.indptr = z["indptr"]
        self.n_nodes = int(z["n_nodes"][0])
        self.n_edges = len(self.link_ids)
        self.overlap_threshold = float(overlap_threshold)

        sorted_keys = (
            np.asarray(self.edge_u, dtype=np.int64)[self.sort_order]
            * np.int64(self.n_nodes)
            + np.asarray(self.edge_v, dtype=np.int64)[self.sort_order]
        )
        self.unique_keys = sorted_keys[self.group_starts]

        physical = np.load(city_dir / "weight_length.npy", mmap_mode="r")
        sorted_physical = np.asarray(physical, dtype=np.float64)[self.sort_order]
        self.pair_physical_length = np.minimum.reduceat(
            sorted_physical, self.group_starts
        )

    def _csr(self, weights: np.ndarray) -> csr_matrix:
        if len(weights) != self.n_edges:
            raise ValueError("weight vector does not match topology")
        sorted_w = np.asarray(weights, dtype=np.float64)[self.sort_order]
        group_w = np.minimum.reduceat(sorted_w, self.group_starts)
        if not np.all(np.isfinite(group_w)):
            raise ValueError("non-finite edge weight")
        if np.any(group_w < 0):
            raise ValueError("Dijkstra requires non-negative edge weights")
        return csr_matrix(
            (group_w, self.indices, self.indptr),
            shape=(self.n_nodes, self.n_nodes),
        )

    @staticmethod
    def _reconstruct_nodes(
        predecessor: np.ndarray, source: int, target: int
    ) -> list[int] | None:
        if source == target:
            return [source]
        nodes = [int(target)]
        cur = int(target)
        guard = 0
        limit = len(predecessor)
        while cur != source:
            prev = int(predecessor[cur])
            if prev == _NO_PREDECESSOR or prev < 0:
                return None
            nodes.append(prev)
            cur = prev
            guard += 1
            if guard > limit:
                return None
        nodes.reverse()
        return nodes

    def _edge_keys(self, nodes: Sequence[int]) -> np.ndarray:
        if len(nodes) < 2:
            return np.empty(0, dtype=np.int64)
        a = np.asarray(nodes[:-1], dtype=np.int64)
        b = np.asarray(nodes[1:], dtype=np.int64)
        return a * np.int64(self.n_nodes) + b

    def _length_overlap_ratio(
        self, actual_nodes: Sequence[int], predicted_nodes: Sequence[int]
    ) -> float:
        actual = self._edge_keys(actual_nodes)
        predicted = self._edge_keys(predicted_nodes)
        if actual.size == 0:
            return 0.0

        apos = np.searchsorted(self.unique_keys, actual)
        if (
            np.any(apos >= len(self.unique_keys))
            or np.any(self.unique_keys[apos] != actual)
        ):
            raise ValueError("actual route contains an edge absent from topology")

        denom = float(self.pair_physical_length[apos].sum())
        if denom <= 0.0:
            return 0.0

        akeys, acount = np.unique(actual, return_counts=True)
        pkeys, pcount = np.unique(predicted, return_counts=True)
        common, ai, pi = np.intersect1d(
            akeys, pkeys, assume_unique=True, return_indices=True
        )
        if common.size == 0:
            return 0.0

        cpos = np.searchsorted(self.unique_keys, common)
        common_len = float(
            (
                self.pair_physical_length[cpos]
                * np.minimum(acount[ai], pcount[pi])
            ).sum()
        )
        return common_len / denom

    def evaluate(self, weights: np.ndarray, orders: Sequence[dict]) -> RouteBatchResult:
        if not orders:
            return RouteBatchResult(0.0, 0, [], [])

        graph = self._csr(weights)

        sources: list[int] = []
        source_to_row: dict[int, int] = {}
        for o in orders:
            source = int(o["start_node"])
            if source not in source_to_row:
                source_to_row[source] = len(sources)
                sources.append(source)

        _, pred = dijkstra(
            graph,
            directed=True,
            indices=np.asarray(sources, dtype=np.int64),
            return_predecessors=True,
        )
        if pred.ndim == 1:
            pred = pred[None, :]

        binary_scores: list[float] = []
        overlap_scores: list[float] = []
        failures = 0

        for o in orders:
            source = int(o["start_node"])
            target = int(o["end_node"])
            row = source_to_row[source]
            predicted = self._reconstruct_nodes(pred[row], source, target)
            if predicted is None:
                failures += 1
                binary_scores.append(0.0)
                overlap_scores.append(0.0)
                continue

            actual = o["route_nodes"]
            overlap = self._length_overlap_ratio(actual, predicted)
            overlap_scores.append(overlap)
            binary_scores.append(
                route_accuracy_from_overlap(
                    overlap, threshold=self.overlap_threshold
                )
            )

        return RouteBatchResult(
            accuracy=float(np.mean(binary_scores)),
            failures=failures,
            per_order_accuracy=binary_scores,
            per_order_overlap=overlap_scores,
        )
