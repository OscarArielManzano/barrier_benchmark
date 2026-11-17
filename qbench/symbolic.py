"""Módulo 4a — Derivación simbólica (sympy) para validar TMM.

Se resuelven simbólicamente las 4 condiciones de continuidad de ψ y ψ' de la
barrera rectangular:
    x<0 : ψ = e^{ikx} + r e^{-ikx}
    0<x<a: ψ = A e^{κx} + B e^{-κx}
    x>a : ψ = t e^{ikx}
y se obtiene t(k, κ, a) en forma cerrada. Evaluando con κ complejo (κ = iq para
E > V0) la misma expresión cubre ambos regímenes.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import sympy as sp

from .config import Barrier


@lru_cache(maxsize=1)
def build_symbolic_t():
    k, kap, a = sp.symbols("k kappa a", positive=True)
    r, A, B, t = sp.symbols("r A B t")
    eqs = [
        sp.Eq(1 + r, A + B),
        sp.Eq(sp.I * k * (1 - r), kap * (A - B)),
        sp.Eq(A * sp.exp(kap * a) + B * sp.exp(-kap * a), t * sp.exp(sp.I * k * a)),
        sp.Eq(kap * (A * sp.exp(kap * a) - B * sp.exp(-kap * a)), sp.I * k * t * sp.exp(sp.I * k * a)),
    ]
    sol = sp.solve(eqs, [r, A, B, t], dict=True)[0]
    t_expr = sp.simplify(sol[t])
    f = sp.lambdify((k, kap, a), t_expr, modules="numpy")
    return t_expr, f


def transmission(E, b: Barrier) -> np.ndarray:
    """T = |t|² evaluando la solución simbólica (E ≠ V0)."""
    _, f = build_symbolic_t()
    E = np.atleast_1d(np.asarray(E, dtype=float))
    k = np.sqrt(E / b.C)
    kap = np.sqrt(((b.V0 - E) / b.C).astype(complex))
    return np.abs(f(k, kap, b.a)) ** 2
