"""Módulo 7 — Gráficas."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import tmm
from .config import Barrier

C_TMM, C_WKB, C_KEM, C_TDSE = "black", "tab:blue", "tab:cyan", "tab:red"
FLOOR = 1e-6  # piso para escalas logarítmicas de error


def _title(b: Barrier, meta: dict, xname: str) -> str:
    parts = []
    if xname != "V0":
        parts.append(f"V0={b.V0:g} eV")
    if xname != "a":
        parts.append(f"a={b.a:g} nm")
    if xname != "E" and "E" in meta:
        parts.append(f"E={meta['E']:g} eV")
    parts.append(f"m={b.m:g} mₑ")
    return ", ".join(parts)


def plot_sweep(exp: dict, path: str, tol: float = 0.05) -> str:
    """Figura de 3 paneles: T (log), error relativo (log) y error en décadas."""
    x, d, meta = exp["x"], exp["data"], exp["meta"]
    b = meta["barrier"]
    fig, ax = plt.subplots(3, 1, figsize=(8.2, 10.5), sharex=True, gridspec_kw={"height_ratios": [1.5, 1.1, 0.9]})

    ax[0].semilogy(x, d["T_tmm"], "-", color=C_TMM, lw=2, label="TMM (referencia exacta)")
    ax[0].semilogy(x, d["T_wkb"], "--", color=C_WKB, lw=1.8, label="WKB (Gamow)")
    ax[0].semilogy(x, d["T_kemble"], ":", color=C_KEM, lw=1.8, label="WKB (Kemble)")
    ax[0].semilogy(x, d["T_tdse"], "o", color=C_TDSE, ms=4.5, mfc="none", label="TDSE (split-operator)")
    ax[0].set_ylabel("Probabilidad de transmisión  T")
    ax[0].set_title(f"{exp['name']}: {_title(b, meta, exp['xname'])}")
    ax[0].legend(loc="best", fontsize=9)
    ax[0].grid(True, which="both", alpha=0.25)

    for key, col, mk, lab in (("wkb", C_WKB, "s--", "WKB (Gamow)"), ("kemble", C_KEM, "^:", "WKB (Kemble)"), ("tdse", C_TDSE, "o-", "TDSE")):
        ax[1].semilogy(x, np.maximum(d[f"err_{key}"], FLOOR), mk, color=col, ms=4, lw=1.2, label=lab)
    ax[1].axhline(tol, color="gray", ls="--", lw=1)
    ax[1].text(x[0], tol * 1.15, f"tolerancia {tol:.0%}", color="gray", fontsize=8)
    ax[1].set_ylabel("Error relativo  |T − T_TMM| / T_TMM")
    ax[1].legend(loc="best", fontsize=9)
    ax[1].grid(True, which="both", alpha=0.25)

    for key, col, mk, lab in (("wkb", C_WKB, "s--", "WKB (Gamow)"), ("kemble", C_KEM, "^:", "WKB (Kemble)"), ("tdse", C_TDSE, "o-", "TDSE")):
        ax[2].semilogy(x, np.maximum(d[f"dec_{key}"], FLOOR), mk, color=col, ms=4, lw=1.2, label=lab)
    ax[2].set_ylabel("Error en décadas  |log₁₀(T/T_TMM)|")
    ax[2].set_xlabel(exp["xlabel"])
    ax[2].grid(True, which="both", alpha=0.25)

    if "vline" in meta:
        for a_ in ax:
            a_.axvline(meta["vline"], color="green", ls="-.", lw=1, alpha=0.7)
        ax[0].text(meta["vline"], ax[0].get_ylim()[0] * 1.5, " " + meta["vline_label"], color="green", fontsize=8, va="bottom")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_cost(exp: dict, path: str) -> str:
    rows, meta = exp["rows"], exp["meta"]
    dxs = sorted({r["dx"] for r in rows}, reverse=True)
    dts = sorted({r["dt_target"] for r in rows}, reverse=True)
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    # (a) error vs dt para cada dx
    for i, dx in enumerate(dxs):
        rr = sorted([r for r in rows if r["dx"] == dx], key=lambda r: r["dt"])
        ax[0].loglog([r["dt"] for r in rr], [max(r["err"], FLOOR) for r in rr], "o-", color=cmap(i / max(1, len(dxs) - 1)), label=f"dx={dx:.4f} nm")
    dtp = np.array(sorted({r["dt"] for r in rows}))
    ax[0].loglog(dtp, 0.3 * (dtp / dtp.max()) ** 1.5 * 1.0, "k:", lw=1, label="guía ∝ dt^1.5")
    ax[0].set_xlabel("dt  [ħ/eV]")
    ax[0].set_ylabel("Error relativo de T")
    ax[0].set_title("(a) Convergencia temporal")
    ax[0].legend(fontsize=8)
    ax[0].grid(True, which="both", alpha=0.25)

    # (b) error vs dx para cada dt
    for j, dt in enumerate(dts):
        rr = sorted([r for r in rows if r["dt_target"] == dt], key=lambda r: r["dx"])
        ax[1].loglog([r["dx"] for r in rr], [max(r["err"], FLOOR) for r in rr], "s-", color=cmap(j / max(1, len(dts) - 1)), label=f"dt={dt:g}")
    ax[1].set_xlabel("dx  [nm]")
    ax[1].set_title("(b) Convergencia espacial")
    ax[1].legend(fontsize=8, ncol=2)
    ax[1].grid(True, which="both", alpha=0.25)

    # (c) frente de Pareto: error vs tiempo de cómputo
    t = np.array([r["runtime"] for r in rows])
    e = np.array([max(r["err"], FLOOR) for r in rows])
    sc_ = ax[2].scatter(t, e, c=[np.log10(r["dx"]) for r in rows], cmap="viridis_r", s=40, edgecolor="k", lw=0.4, label="TDSE (cada punto = (dx, dt))")
    order = np.argsort(t)
    best = np.minimum.accumulate(e[order])
    ax[2].step(t[order], best, where="post", color="gray", lw=1, label="frente de Pareto")
    ax[2].scatter([meta["t_wkb"]], [max(meta["err_wkb"], FLOOR)], marker="*", s=160, color=C_WKB, edgecolor="k", label="WKB", zorder=5)
    ax[2].axvline(meta["t_tmm"], color="black", lw=1.5, label="TMM (error 0 por construcción)")
    ax[2].set_xscale("log")
    ax[2].set_yscale("log")
    ax[2].set_xlabel("Tiempo de cómputo por punto [s]")
    ax[2].set_title("(c) Costo vs precisión")
    ax[2].legend(fontsize=8, loc="lower left")
    ax[2].grid(True, which="both", alpha=0.25)
    cb = fig.colorbar(sc_, ax=ax[2])
    cb.set_label("log₁₀ dx")

    b = meta["barrier"]
    fig.suptitle(f"Exp.4 — Costo computacional vs precisión (E={meta['E']:g} eV, V0={b.V0:g} eV, a={b.a:g} nm)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_band(r, b: Barrier, path: str) -> str:
    """Una sola corrida TDSE reproduce T(E) en toda la banda espectral del paquete."""
    E = b.C * r.k_band**2
    ref = tmm.transmission(E, b)
    fig, ax = plt.subplots(2, 1, figsize=(7, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax[0].semilogy(E, ref, "k-", lw=2, label="TMM")
    ax[0].semilogy(E, r.T_band, "r.", ms=4, label="TDSE (cociente espectral, 1 corrida)")
    ax[0].axvline(r.E0, color="gray", ls=":")
    ax[0].set_ylabel("T(E)")
    ax[0].legend()
    ax[0].grid(True, which="both", alpha=0.25)
    ax[1].plot(E, (r.T_band - ref) / ref * 100, "r.-", ms=3, lw=0.6)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_ylabel("error rel. [%]")
    ax[1].set_xlabel("E [eV]")
    ax[1].grid(True, alpha=0.25)
    fig.suptitle("Validación: una corrida TDSE cubre una banda de energías")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
