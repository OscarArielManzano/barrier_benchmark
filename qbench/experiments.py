"""Módulo 6 — Experimentos del benchmark.

Exp. 1  T vs E      (m, V0, a fijos)
Exp. 2  T vs a      (m, V0, E fijos)
Exp. 3  T vs V0     (m, E, a fijos)
Exp. 4  costo vs precisión del TDSE (barrido de dx y dt) y comparación de
        costos con WKB y TMM.

Referencia: TMM (exacta para la barrera rectangular, validada en validation.py).
Errores: relativo |T - T_ref|/T_ref  y en décadas |log10(T/T_ref)|.
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import metrics, tdse, tmm, wkb
from .config import DEFAULT_BARRIER, Barrier


# ----------------------------------------------------------------------------
# Núcleo: evalúa los tres métodos en una lista de puntos (E, barrera)
# ----------------------------------------------------------------------------
def evaluate_points(
    E_list: Sequence[float],
    barriers: Sequence[Barrier],
    cfg: tdse.TDSEConfig,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    n = len(E_list)
    keys = ["T_tmm", "T_wkb", "T_kemble", "T_tdse", "t_tmm", "t_wkb", "t_tdse", "N", "nsteps"]
    out = {k: np.zeros(n) for k in keys}
    for i, (E, b) in enumerate(zip(E_list, barriers)):
        out["T_tmm"][i] = tmm.transmission(E, b)[0]
        out["T_wkb"][i] = wkb.transmission(E, b, "gamow")[0]
        out["T_kemble"][i] = wkb.transmission(E, b, "kemble")[0]
        out["t_tmm"][i] = metrics.time_call(lambda: tmm.transmission(E, b), repeat=3, number=20)
        out["t_wkb"][i] = metrics.time_call(lambda: wkb.transmission(E, b), repeat=3, number=200)
        r = tdse.run(E, b, cfg)
        out["T_tdse"][i] = r.T
        out["t_tdse"][i] = r.runtime
        out["N"][i] = r.N
        out["nsteps"][i] = r.nsteps
        if verbose:
            print(
                f"    [{i + 1:>3}/{n}] E={E:.3f} V0={b.V0:.2f} a={b.a:.3f}  "
                f"T_TMM={out['T_tmm'][i]:.3e}  T_WKB={out['T_wkb'][i]:.3e}  "
                f"T_TDSE={r.T:.3e}  ({r.runtime:.1f} s)",
                flush=True,
            )
    ref = out["T_tmm"]
    out["err_wkb"] = metrics.rel_error(out["T_wkb"], ref)
    out["err_kemble"] = metrics.rel_error(out["T_kemble"], ref)
    out["err_tdse"] = metrics.rel_error(out["T_tdse"], ref)
    out["dec_wkb"] = metrics.log10_error(out["T_wkb"], ref)
    out["dec_kemble"] = metrics.log10_error(out["T_kemble"], ref)
    out["dec_tdse"] = metrics.log10_error(out["T_tdse"], ref)
    return out


def _pack(name: str, xname: str, xlabel: str, x, out, meta: dict) -> dict:
    return {"name": name, "xname": xname, "xlabel": xlabel, "x": np.asarray(x, float), "data": out, "meta": meta}


# ----------------------------------------------------------------------------
# Experimentos 1-3
# ----------------------------------------------------------------------------
def exp_energy(b: Barrier = DEFAULT_BARRIER, cfg=tdse.TDSEConfig(), n: int = 32, verbose=True) -> dict:
    E = np.linspace(0.10, 1.60, n) * b.V0
    if verbose:
        print(f"  Exp.1  T vs E  ({b}), {n} puntos")
    out = evaluate_points(E, [b] * n, cfg, verbose)
    return _pack("exp1_energia", "E", "Energía E [eV]", E, out, {"barrier": b, "vline": b.V0, "vline_label": "E = V0"})


def exp_width(b: Barrier = DEFAULT_BARRIER, E: float = 0.5, cfg=tdse.TDSEConfig(), n: int = 24, verbose=True) -> dict:
    a = np.linspace(0.10, 1.50, n)
    if verbose:
        print(f"  Exp.2  T vs a  (E={E} eV, V0={b.V0} eV), {n} puntos")
    out = evaluate_points([E] * n, [b.replace(a=float(ai)) for ai in a], cfg, verbose)
    return _pack("exp2_ancho", "a", "Ancho de barrera a [nm]", a, out, {"barrier": b, "E": E})


def exp_height(b: Barrier = DEFAULT_BARRIER, E: float = 0.5, cfg=tdse.TDSEConfig(), n: int = 24, verbose=True) -> dict:
    V0 = np.linspace(0.6, 3.0, n)
    if verbose:
        print(f"  Exp.3  T vs V0 (E={E} eV, a={b.a} nm), {n} puntos")
    out = evaluate_points([E] * n, [b.replace(V0=float(v)) for v in V0], cfg, verbose)
    return _pack("exp3_altura", "V0", "Altura de barrera V0 [eV]", V0, out, {"barrier": b, "E": E, "vline": E, "vline_label": "V0 = E"})


# ----------------------------------------------------------------------------
# Experimento 4: costo computacional vs precisión
# ----------------------------------------------------------------------------
def exp_cost(
    b: Barrier = DEFAULT_BARRIER,
    E: float = 0.5,
    dx_list: Sequence[float] = (0.125, 0.0625, 0.03125, 0.015625),
    dt_list: Sequence[float] = (0.4, 0.2, 0.1, 0.05, 0.025, 0.0125),
    verbose=True,
) -> dict:
    T_ref = float(tmm.transmission(E, b)[0])
    rows: List[dict] = []
    if verbose:
        print(f"  Exp.4  costo vs precisión (E={E} eV, {b}); T_ref={T_ref:.5e}")
    for dx in dx_list:
        for dt in dt_list:
            r = tdse.run(E, b, tdse.TDSEConfig(dx=dx, dt=dt))
            err = float(metrics.rel_error(r.T, T_ref))
            rows.append(dict(dx=r.dx, dt=r.dt, dt_target=dt, N=r.N, nsteps=r.nsteps, runtime=r.runtime, T=r.T, err=err))
            if verbose:
                print(f"    dx={r.dx:.4f} dt={dt:<6} N={r.N:<6} pasos={r.nsteps:<6} t={r.runtime:6.2f} s  err={err:.2e}", flush=True)
    t_tmm = metrics.time_call(lambda: tmm.transmission(E, b), repeat=5, number=50)
    t_wkb = metrics.time_call(lambda: wkb.transmission(E, b), repeat=5, number=500)
    T_w = float(wkb.transmission(E, b)[0])
    meta = dict(E=E, T_ref=T_ref, t_tmm=t_tmm, t_wkb=t_wkb, err_wkb=float(metrics.rel_error(T_w, T_ref)), barrier=b)
    return {"name": "exp4_costo", "rows": rows, "meta": meta}


# ----------------------------------------------------------------------------
# Reportes y persistencia
# ----------------------------------------------------------------------------
def report(exp: dict, tol: float = 0.05) -> str:
    x, d, xn = exp["x"], exp["data"], exp["xname"]
    lines = [f"== {exp['name']} =="]
    for key, label in (("wkb", "WKB (Gamow)"), ("kemble", "WKB (Kemble)"), ("tdse", "TDSE (split-operator)")):
        s = metrics.summary(d[f"err_{key}"])
        iv = metrics.within_tolerance(x, d[f"err_{key}"], tol)
        iv2 = metrics.within_tolerance(x, d[f"dec_{key}"], np.log10(2.0))
        lines.append(
            f"  {label:<24s} err.rel: máx={s['max']:.2e} media={s['mean']:.2e} rms={s['rms']:.2e} | "
            f"décadas máx={np.max(d['dec_' + key]):.3f}\n"
            f"  {'':<24s} err ≤ {tol:.0%} en {xn} ∈ {metrics.fmt_intervals(iv)} | "
            f"factor ≤ 2 en {xn} ∈ {metrics.fmt_intervals(iv2)}"
        )
    lines.append(
        f"  Costo medio/punto: TMM={np.mean(d['t_tmm']) * 1e6:.1f} µs, WKB={np.mean(d['t_wkb']) * 1e6:.1f} µs, "
        f"TDSE={np.mean(d['t_tdse']):.2f} s"
    )
    return "\n".join(lines)


def save_csv(exp: dict, outdir: str) -> str:
    path = os.path.join(outdir, f"{exp['name']}.csv")
    if "rows" in exp:
        rows = exp["rows"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        return path
    d = exp["data"]
    cols = [k for k in d.keys()]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([exp["xname"]] + cols)
        for i in range(len(exp["x"])):
            w.writerow([f"{exp['x'][i]:.8g}"] + [f"{d[c][i]:.8e}" for c in cols])
    return path
