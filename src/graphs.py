from __future__ import annotations

import numpy as np


def adjacency_matrix(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    kind = kind.lower()
    if kind == "ring":
        return ring_graph(n)
    if kind == "small_world":
        return small_world_graph(n, k=4, p=0.12, rng=rng)
    if kind == "modular":
        return modular_graph(n, modules=4, p_in=0.32, p_out=0.025, rng=rng)
    if kind == "scale_free":
        return scale_free_graph(n, m=3, rng=rng)
    raise ValueError(f"unknown graph kind: {kind}")


def ring_graph(n: int) -> np.ndarray:
    a = np.zeros((n, n), dtype=float)
    for i in range(n):
        for step in (1, 2):
            a[i, (i + step) % n] = 1.0
            a[(i + step) % n, i] = 1.0
    return a


def small_world_graph(n: int, k: int, p: float, rng: np.random.Generator) -> np.ndarray:
    a = np.zeros((n, n), dtype=float)
    half = k // 2
    for i in range(n):
        for d in range(1, half + 1):
            j = (i + d) % n
            a[i, j] = a[j, i] = 1.0
    for i in range(n):
        for j in range(i + 1, n):
            if a[i, j] > 0 and rng.random() < p:
                candidates = np.flatnonzero((a[i] == 0) & (np.arange(n) != i))
                if len(candidates) > 0:
                    new_j = int(rng.choice(candidates))
                    a[i, j] = a[j, i] = 0.0
                    a[i, new_j] = a[new_j, i] = 1.0
    return ensure_connected(a)


def modular_graph(
    n: int,
    modules: int,
    p_in: float,
    p_out: float,
    rng: np.random.Generator,
) -> np.ndarray:
    labels = np.arange(n) * modules // n
    a = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            p = p_in if labels[i] == labels[j] else p_out
            if rng.random() < p:
                a[i, j] = a[j, i] = 1.0
    for m in range(modules - 1):
        left = np.flatnonzero(labels == m)
        right = np.flatnonzero(labels == m + 1)
        i = int(rng.choice(left))
        j = int(rng.choice(right))
        a[i, j] = a[j, i] = 1.0
    return ensure_connected(a)


def scale_free_graph(n: int, m: int, rng: np.random.Generator) -> np.ndarray:
    a = np.zeros((n, n), dtype=float)
    initial = max(m + 1, 4)
    for i in range(initial):
        for j in range(i + 1, initial):
            a[i, j] = a[j, i] = 1.0
    degree = a.sum(axis=1)
    for new in range(initial, n):
        probs = degree[:new] + 1e-6
        probs = probs / probs.sum()
        targets = rng.choice(np.arange(new), size=min(m, new), replace=False, p=probs)
        for t in targets:
            a[new, t] = a[t, new] = 1.0
        degree = a.sum(axis=1)
    return ensure_connected(a)


def ensure_connected(a: np.ndarray) -> np.ndarray:
    n = a.shape[0]
    components = connected_components(a)
    if len(components) <= 1:
        return a
    for c1, c2 in zip(components[:-1], components[1:]):
        i = int(c1[0])
        j = int(c2[0])
        a[i, j] = a[j, i] = 1.0
    return a


def connected_components(a: np.ndarray) -> list[np.ndarray]:
    n = a.shape[0]
    seen = np.zeros(n, dtype=bool)
    comps = []
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        comp = []
        while stack:
            i = stack.pop()
            comp.append(i)
            for j in np.flatnonzero(a[i] > 0):
                if not seen[j]:
                    seen[j] = True
                    stack.append(int(j))
        comps.append(np.asarray(comp, dtype=int))
    return comps


def laplacian(a: np.ndarray, normalized: bool = False) -> np.ndarray:
    d = a.sum(axis=1)
    l = np.diag(d) - a
    if not normalized:
        return l
    inv = np.zeros_like(d)
    nz = d > 0
    inv[nz] = 1.0 / np.sqrt(d[nz])
    return np.diag(inv) @ l @ np.diag(inv)


def eigensystem(l: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vals, vecs = np.linalg.eigh(l)
    order = np.argsort(vals)
    return vals[order], vecs[:, order]

