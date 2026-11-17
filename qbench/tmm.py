"""Módulo 1 — TMM (Transfer Matrix Method).

Para un potencial constante a trozos la ecuación de Schrödinger estacionaria
se resuelve *exactamente* capa por capa; la continuidad de ψ y ψ' en cada
interfaz se codifica en matrices 2×2. Por eso TMM es la **solución de
referencia** del benchmark (se valida contra la fórmula analítica y contra
una derivación simbólica con sympy, ver `validation.py`).

Formulación usada (misma masa en todas las regiones):
    capa j (número de onda k_j, espesor d_j):
        (ψ, ψ')_derecha = W_j (ψ, ψ')_izquierda
        W_j = [[cos k d,      sin(k d)/k],
               [-k sin k d,   cos k d   ]]          (k complejo si E < V)
    matriz total  W = W_n ··· W_1,   det W = 1
    (A_L, B_L)ᵀ = D_L⁻¹ W⁻¹ D_R (A_R, 0)ᵀ = M (A_R, 0)ᵀ ,   D = [[1, 1],[ik, -ik]]
    T = 1/|M₀₀|² ,   R = |M₁₀/M₀₀|²
"""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np

from .config import Barrier


def _layer_matrix(k: complex, d: float) -> np.ndarray:
    kd = k * d
    s_over_k = d if abs(kd) < 1e-9 else np.sin(kd) / k
    return np.array([[np.cos(kd), s_over_k], [-k * np.sin(kd), np.cos(kd)]], dtype=complex)


def transmission_layers(
    E: Iterable[float] | float,
    layers: Sequence[Tuple[float, float]],
    C: float,
    return_R: bool = False,
):
    """T(E) (y opcionalmente R(E)) para un apilado de capas [(V_j, d_j), ...]
    entre dos guías semiinfinitas con V = 0."""
    E = np.atleast_1d(np.asarray(E, dtype=float))
    if np.any(E <= 0):
        raise ValueError("TMM requiere E > 0 (guías con V=0).")
    T = np.empty_like(E)
    R = np.empty_like(E)
    for i, e in enumerate(E):
        kl = np.sqrt(e / C)
        W = np.eye(2, dtype=complex)
        for V, d in layers:
            k = np.sqrt(complex((e - V) / C))  # +iκ cuando e < V
            W = _layer_matrix(k, d) @ W
        Winv = np.array([[W[1, 1], -W[0, 1]], [-W[1, 0], W[0, 0]]])  # det W = 1
        D = np.array([[1, 1], [1j * kl, -1j * kl]], dtype=complex)
        M = np.linalg.inv(D) @ Winv @ D
        T[i] = 1.0 / abs(M[0, 0]) ** 2
        R[i] = abs(M[1, 0] / M[0, 0]) ** 2
    return (T, R) if return_R else T


def transmission(E, b: Barrier, return_R: bool = False):
    """Transmisión de la barrera rectangular mediante TMM."""
    return transmission_layers(E, [(b.V0, b.a)], b.C, return_R=return_R)


def analytic_rectangular(E, b: Barrier) -> np.ndarray:
    """Fórmula analítica clásica (libro de texto) — sólo para validación."""
    E = np.atleast_1d(np.asarray(E, dtype=float))
    V0, a, C = b.V0, b.a, b.C
    d = V0 - E
    T = np.empty_like(E)
    below = d > 1e-12 * V0
    above = d < -1e-12 * V0
    eq = ~(below | above)
    kap = np.sqrt(np.clip(d, 0, None) / C)
    q = np.sqrt(np.clip(-d, 0, None) / C)
    T[below] = 1.0 / (1.0 + V0**2 * np.sinh(kap[below] * a) ** 2 / (4 * E[below] * d[below]))
    T[above] = 1.0 / (1.0 + V0**2 * np.sin(q[above] * a) ** 2 / (4 * E[above] * (-d[above])))
    T[eq] = 1.0 / (1.0 + V0 * a**2 / (4 * C))  # límite E → V0
    return T
