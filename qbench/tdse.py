"""Módulo 3 — TDSE (ecuación de Schrödinger dependiente del tiempo).

Método: split-operator de Strang con FFT (espectral en x, 2.º orden en t)

    ψ(t+dt) = e^{-iV dt/2} · F⁻¹ e^{-iC k² dt} F · e^{-iV dt/2} ψ(t)

Se propaga un paquete gaussiano  ψ0 = N exp(-(x-x0)²/4σx²) e^{i k0 x}  contra
la barrera. Como el paquete tiene dispersión en energía, la transmisión se
extrae **espectralmente**:

    T(k) = |F[ψ(t_f)·Θ(x > x_cut)](k)|² / |F[ψ(0)](k)|²

(cociente entre el espectro transmitido y el incidente; ambos lados tienen
V = 0, por lo que el cociente de densidades en k es el cociente de flujos).
El valor reportado es T(k0) obtenido con un ajuste polinómico local de ln T(k)
(suaviza el rizado que introduce el splitting con un potencial discontinuo); la
banda completa T(k) queda disponible como diagnóstico (una sola corrida cubre
k0 ± 2σk).

Detalles numéricos que importan para la precisión
-------------------------------------------------
* dx se ajusta para que el ancho a sea un número *par* de celdas y los bordes
  de la barrera caigan exactamente entre puntos de malla (sin error de ancho).
* El paquete empieza a `n_sigma_start`·σx de la barrera para que su cola
  gaussiana no solape con ella (error ∼ e^{-n²/2}).
* Capa absorbente suave en los bordes: evita reinyección periódica.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy import fft as sfft

from .config import Barrier


@dataclass(frozen=True)
class TDSEConfig:
    dx: float = 0.02              # paso espacial objetivo [nm]
    dt: float = 0.025             # paso temporal máximo [ħ/eV]
    sigma_k_rel: float = 0.1      # σk = sigma_k_rel · k0 ...
    sigma_k_min: float = 0.05     # ... acotado a [min, max] [nm⁻¹]
    sigma_k_max: float = 0.5
    n_sigma_start: float = 8.0    # distancia inicial paquete–barrera en σx
    n_out: float = 5.0            # el paquete transmitido debe quedar a n_out·σ(t) de x_cut
    pad: float = 3.0              # holgura extra [nm]
    w_abs: float = 6.0            # espesor del absorbente [nm]
    gamma_abs: float = 2.0        # tasa máxima de absorción [eV]
    cut: float = 2.0              # x_cut = a/2 + cut [nm] (fuera del campo cercano)
    band_nsigma: float = 2.0      # semiancho de la banda espectral en σk
    fit_nsigma: float = 1.5       # ventana (en σk) del ajuste local de ln T(k)
    fit_deg: int = 3              # grado del polinomio del ajuste


@dataclass
class TDSEResult:
    E0: float
    T: float              # T(k0): ajuste local de ln T(k) (observable principal)
    T_raw: float          # T(k0) por interpolación directa del cociente (ruidoso)
    P_trans: float        # probabilidad total transmitida (x > x_cut)
    P_refl: float         # probabilidad total reflejada (x < -a/2)
    norm: float           # norma final (≈1; <1 sólo por el absorbente)
    runtime: float        # tiempo de pared [s]
    N: int
    nsteps: int
    dx: float
    dt: float
    sigma_k: float
    k_band: np.ndarray
    T_band: np.ndarray


def run(E0: float, b: Barrier, cfg: TDSEConfig = TDSEConfig()) -> TDSEResult:
    t_start = perf_counter()
    C = b.C
    k0 = float(np.sqrt(E0 / C))
    sk = float(np.clip(cfg.sigma_k_rel * k0, cfg.sigma_k_min, cfg.sigma_k_max))
    sk = min(sk, k0 / 4.0)
    sx = 1.0 / (2.0 * sk)

    # --- geometría y tiempo ----------------------------------------------------
    # El paquete se dispersa: σ(t) = σx·sqrt(1 + (C t/σx²)²). Se elige el tiempo
    # final de modo que el paquete transmitido quede COMPLETO a la derecha de
    # x_cut (n_out·σ(t_f) de margen); si no, el truncamiento contamina T(k).
    v0 = 2.0 * C * k0                                   # velocidad de grupo
    x0 = -(b.a / 2 + cfg.n_sigma_start * sx + cfg.pad)  # centro inicial
    D = cfg.cut + cfg.n_out * sx
    for _ in range(60):                                 # punto fijo para D
        t_f = (b.a / 2 + D - x0) / v0
        sig_f = sx * np.sqrt(1.0 + (C * t_f / sx**2) ** 2)
        D_new = cfg.cut + cfg.n_out * sig_f
        if abs(D_new - D) < 1e-6:
            break
        D = D_new
    t_final = (b.a / 2 + D - x0) / v0
    sig_f = sx * np.sqrt(1.0 + (C * t_final / sx**2) ** 2)
    # T(k) crece con k: el paquete transmitido se adelanta y ensancha -> margen extra
    half = max(abs(x0) + 6.0 * sx, b.a / 2 + D + (cfg.n_out + 3.0) * sig_f) + cfg.w_abs

    # --- malla (barrera centrada en 0, bordes entre celdas) --------------------
    n_b = max(2, 2 * int(round(b.a / (2 * cfg.dx))))
    dx = b.a / n_b
    N = 2 * int(np.ceil(half / dx))
    N = sfft.next_fast_len(N)               # tamaños "rápidos" para la FFT
    while N % 2:                            # N par -> malla simétrica respecto a 0
        N = sfft.next_fast_len(N + 1)
    x = (np.arange(N) + 0.5 - N / 2) * dx
    V = np.where(np.abs(x) < b.a / 2, b.V0, 0.0)

    nsteps = max(1, int(np.ceil(t_final / cfg.dt)))
    dt = t_final / nsteps

    # --- absorbente y propagadores ---------------------------------------------
    x_abs = N / 2 * dx - cfg.w_abs
    s = np.clip((np.abs(x) - x_abs) / cfg.w_abs, 0.0, 1.0)
    damp = np.exp(-cfg.gamma_abs * s**2 * dt)
    kk = 2.0 * np.pi * sfft.fftfreq(N, d=dx)
    pk = np.exp(-1j * C * kk**2 * dt)
    pv_half = np.exp(-0.5j * V * dt)
    pv_full = np.exp(-1j * V * dt) * damp

    # --- estado inicial ----------------------------------------------------------
    psi0 = np.exp(-((x - x0) ** 2) / (4 * sx**2)) * np.exp(1j * k0 * x)
    psi0 /= np.sqrt(np.sum(np.abs(psi0) ** 2) * dx)

    # --- evolución (Strang: V/2 · [T · V]^(n-1) · T · V/2) ----------------------
    psi = psi0 * pv_half
    for n in range(nsteps):
        psi = sfft.ifft(pk * sfft.fft(psi))
        psi *= pv_full if n < nsteps - 1 else pv_half

    # --- análisis espectral ------------------------------------------------------
    x_cut = b.a / 2 + cfg.cut
    right = x > x_cut
    Phi0 = sfft.fft(psi0)
    PhiT = sfft.fft(psi * right)
    sel = (kk > 0) & (np.abs(kk - k0) <= cfg.band_nsigma * sk)
    order = np.argsort(kk[sel])
    kb = kk[sel][order]
    p0 = (np.abs(Phi0) ** 2)[sel][order]
    pT = (np.abs(PhiT) ** 2)[sel][order]
    Tb = pT / p0
    T_raw = float(np.interp(k0, kb, Tb))
    # El error de splitting con potencial discontinuo produce un rizado de pequeña
    # amplitud en T(k); un ajuste polinómico local de ln T lo promedia.
    win = np.abs(kb - k0) <= cfg.fit_nsigma * sk
    coef = np.polyfit(kb[win] - k0, np.log(np.maximum(Tb[win], 1e-300)), cfg.fit_deg)
    T0 = float(np.exp(coef[-1]))

    dens = np.abs(psi) ** 2
    return TDSEResult(
        E0=E0,
        T=T0,
        T_raw=T_raw,
        P_trans=float(np.sum(dens[right]) * dx),
        P_refl=float(np.sum(dens[x < -b.a / 2]) * dx),
        norm=float(np.sum(dens) * dx),
        runtime=perf_counter() - t_start,
        N=N,
        nsteps=nsteps,
        dx=dx,
        dt=dt,
        sigma_k=sk,
        k_band=kb,
        T_band=Tb,
    )
