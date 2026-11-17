"""Módulo 2 — Aproximación WKB (semiclásica).

Para E < V0 la transmisión se estima con la integral de acción en la región
clásicamente prohibida:

        θ = ∫ κ(x) dx ,   κ(x) = sqrt(2m (V(x) - E)) / ħ
        T_WKB = exp(-2 θ)                       (variante 'gamow')

Para la barrera rectangular θ = κ·a. Se incluye también la variante 'kemble'
    T = 1 / (1 + exp(2θ))
que incorpora la conexión en los puntos de retorno y evita T > 1.

Para E ≥ V0 no hay región prohibida ni puntos de retorno: WKB predice T = 1
(no describe la reflexión sobre los escalones del potencial). Esa es una de las
limitaciones que el benchmark cuantifica.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.integrate import trapezoid

from .config import Barrier


def exponent(E, b: Barrier) -> np.ndarray:
    """θ = κ a para la barrera rectangular (0 si E ≥ V0)."""
    return b.kappa(E) * b.a


def transmission(E, b: Barrier, variant: str = "gamow") -> np.ndarray:
    E = np.atleast_1d(np.asarray(E, dtype=float))
    th = exponent(E, b)
    with np.errstate(over="ignore"):
        if variant == "gamow":
            T = np.exp(-2.0 * th)
        elif variant == "kemble":
            T = 1.0 / (1.0 + np.exp(2.0 * th))
        else:
            raise ValueError("variant debe ser 'gamow' o 'kemble'")
    T = np.where(E >= b.V0, 1.0, T)
    return T


def exponent_numeric(E: float, V: Callable[[np.ndarray], np.ndarray], C: float, x: np.ndarray) -> float:
    """θ para un potencial genérico V(x) sobre una malla x (extensión futura:
    barreras trapezoidales, parabólicas, etc.)."""
    kap = np.sqrt(np.clip(V(x) - E, 0.0, None) / C)
    return float(trapezoid(kap, x))
