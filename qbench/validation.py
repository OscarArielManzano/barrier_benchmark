"""Módulo 5 — Validación de las implementaciones antes de hacer el benchmark."""
from __future__ import annotations

import os

import numpy as np

from . import metrics, symbolic, tdse, tmm, wkb
from .config import DEFAULT_BARRIER, Barrier


def _line(ok: bool, name: str, value: str) -> str:
    return f"  [{'OK' if ok else 'FALLA'}] {name:<52s} {value}"


def run_validation(b: Barrier = DEFAULT_BARRIER, outdir: str | None = None, verbose: bool = True) -> dict:
    res: dict = {}
    lines = [f"Validación con {b}"]

    # 1) TMM vs fórmula analítica ------------------------------------------------
    E = np.linspace(0.02, 3.0, 400)
    E = np.append(E, b.V0)  # incluye el punto singular E = V0
    T_tmm = tmm.transmission(E, b)
    T_an = tmm.analytic_rectangular(E, b)
    d1 = float(np.max(metrics.rel_error(T_tmm, T_an)))
    res["tmm_vs_analytic"] = d1
    lines.append(_line(d1 < 1e-9, "TMM vs fórmula analítica (máx. err. rel.)", f"{d1:.2e}"))

    # 2) TMM vs derivación simbólica (sympy) ------------------------------------
    Es = np.array([0.05, 0.2, 0.5, 0.9, 1.1, 1.5, 2.5])
    d2 = float(np.max(metrics.rel_error(tmm.transmission(Es, b), symbolic.transmission(Es, b))))
    res["tmm_vs_sympy"] = d2
    lines.append(_line(d2 < 1e-9, "TMM vs solución simbólica sympy (máx. err. rel.)", f"{d2:.2e}"))

    # 3) Conservación de probabilidad T + R = 1 -----------------------------------
    T, R = tmm.transmission(E, b, return_R=True)
    d3 = float(np.max(np.abs(T + R - 1)))
    res["TR_conservation"] = d3
    lines.append(_line(d3 < 1e-10, "Conservación T + R = 1 en TMM (máx. |T+R-1|)", f"{d3:.2e}"))

    # 4) WKB en el límite de barrera gruesa ---------------------------------------
    bt = b.replace(a=3.0)
    Et = 0.5 * bt.V0
    ratio = float(tmm.transmission(Et, bt)[0] / wkb.transmission(Et, bt)[0])
    expected = 16 * Et * (bt.V0 - Et) / bt.V0**2
    d4 = abs(ratio / expected - 1)
    res["wkb_thick_limit"] = d4
    lines.append(_line(d4 < 1e-6, "T_exacta/T_WKB → 16E(V0-E)/V0² (barrera gruesa)", f"{ratio:.4f} vs {expected:.4f}"))

    # 5) TDSE: banda espectral vs TMM y conservación de norma ---------------------
    r = tdse.run(0.5 * b.V0, b)
    Eb = b.C * r.k_band**2
    inner = np.abs(r.k_band - np.sqrt(0.5 * b.V0 / b.C)) <= 1.0 * r.sigma_k  # ±1σk
    err_band = metrics.rel_error(r.T_band, tmm.transmission(Eb, b))
    d5 = float(np.max(err_band[inner]))
    res["tdse_band_err"] = d5
    res["tdse_norm"] = r.norm
    lines.append(_line(d5 < 1e-2, "TDSE T(k) vs TMM en k0±σk (máx. err. rel.)", f"{d5:.2e}"))
    lines.append(_line(abs(r.norm - 1) < 1e-3, "TDSE conserva la norma (|norma-1|)", f"{abs(r.norm - 1):.2e}"))
    d6 = abs(r.P_trans + r.P_refl - 1)
    lines.append(_line(d6 < 1e-3, "TDSE P_trans + P_refl ≈ 1", f"{d6:.2e}"))
    lines.append(f"  TDSE: N={r.N}, pasos={r.nsteps}, dx={r.dx:.4g} nm, dt={r.dt:.3g}, t={r.runtime:.2f} s")

    if outdir:
        from .plotting import plot_band

        plot_band(r, b, os.path.join(outdir, "00_validacion_banda_tdse.png"))

    if verbose:
        print("\n".join(lines))
    res["lines"] = lines
    return res
