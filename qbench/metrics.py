"""Módulo 4b — Métricas de error y de costo computacional."""
from __future__ import annotations

from time import perf_counter
from typing import Callable, Dict, List, Tuple

import numpy as np

_TINY = 1e-300


def rel_error(T, T_ref) -> np.ndarray:
    """Error relativo |T - T_ref| / T_ref."""
    T = np.asarray(T, dtype=float)
    T_ref = np.asarray(T_ref, dtype=float)
    return np.abs(T - T_ref) / np.maximum(T_ref, _TINY)


def log10_error(T, T_ref) -> np.ndarray:
    """|log10(T/T_ref)|: error en órdenes de magnitud (útil en tunelamiento,
    donde T abarca muchas décadas)."""
    T = np.maximum(np.asarray(T, dtype=float), _TINY)
    T_ref = np.maximum(np.asarray(T_ref, dtype=float), _TINY)
    return np.abs(np.log10(T / T_ref))


def summary(err: np.ndarray) -> Dict[str, float]:
    err = np.asarray(err, dtype=float)
    return {
        "max": float(np.max(err)),
        "mean": float(np.mean(err)),
        "median": float(np.median(err)),
        "rms": float(np.sqrt(np.mean(err**2))),
    }


def within_tolerance(x, err, tol: float) -> List[Tuple[float, float]]:
    """Intervalos [x_i, x_j] (extremos = puntos muestreados) donde err ≤ tol."""
    x = np.asarray(x, dtype=float)
    ok = np.asarray(err) <= tol
    out, start = [], None
    for i, flag in enumerate(ok):
        if flag and start is None:
            start = i
        if start is not None and (not flag or i == len(ok) - 1):
            end = i if flag else i - 1
            out.append((float(x[start]), float(x[end])))
            start = None
    return out


def fmt_intervals(iv: List[Tuple[float, float]], unit: str = "") -> str:
    if not iv:
        return "ninguno"
    return "; ".join(f"[{lo:.3g}, {hi:.3g}]{unit}" for lo, hi in iv)


def time_call(fn: Callable[[], object], repeat: int = 5, number: int = 10) -> float:
    """Mejor tiempo por llamada [s] (min sobre `repeat` bloques de `number`)."""
    best = np.inf
    for _ in range(repeat):
        t0 = perf_counter()
        for _ in range(number):
            fn()
        best = min(best, (perf_counter() - t0) / number)
    return float(best)
