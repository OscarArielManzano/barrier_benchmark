"""Módulo 0 — Unidades y definición del sistema físico.

Unidades internas del framework
-------------------------------
    energía  : eV
    longitud : nm
    tiempo   : ħ/eV   (por tanto ħ = 1 en las ecuaciones evolutivas)

Con esto la ecuación de Schrödinger 1D queda

    i ∂ψ/∂t = [ -C ∂²/∂x² + V(x) ] ψ ,     C = ħ²/(2m)   [eV·nm²]

y para el electrón libre  C = ħ²/(2 m_e) ≈ 0.0381 eV·nm².
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy import constants as sc

# ħ²/(2 m_e) en eV·nm²  (J·m² -> eV·m² -> eV·nm²)
HBAR2_2ME: float = sc.hbar**2 / (2.0 * sc.m_e) / sc.e * 1e18


@dataclass(frozen=True)
class Barrier:
    """Barrera rectangular  V(x) = V0  para |x| < a/2,  0 en otro caso.

    Parameters
    ----------
    V0 : altura de la barrera [eV]
    a  : ancho de la barrera [nm]
    m  : masa de la partícula en unidades de la masa del electrón
    """

    V0: float = 1.0
    a: float = 0.5
    m: float = 1.0

    @property
    def C(self) -> float:
        """ħ²/(2m) en eV·nm²."""
        return HBAR2_2ME / self.m

    def k(self, E):
        """Número de onda libre k = sqrt(E/C) [nm⁻¹] (E ≥ 0)."""
        return np.sqrt(np.asarray(E, dtype=float) / self.C)

    def kappa(self, E):
        """Constante de decaimiento κ = sqrt((V0-E)/C) (0 si E ≥ V0)."""
        return np.sqrt(np.clip(self.V0 - np.asarray(E, dtype=float), 0.0, None) / self.C)

    def replace(self, **kw) -> "Barrier":
        return replace(self, **kw)

    def __str__(self) -> str:  # pragma: no cover
        return f"Barrier(V0={self.V0} eV, a={self.a} nm, m={self.m} m_e)"


DEFAULT_BARRIER = Barrier(V0=1.0, a=0.5, m=1.0)
