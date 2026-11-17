#!/usr/bin/env python3
"""Ejecuta el benchmark WKB / TMM / TDSE para la barrera rectangular 1D.

Uso:
    python run_benchmark.py                 # todo (validación + exp. 1-4)
    python run_benchmark.py --exp energy    # sólo T vs E
    python run_benchmark.py --exp validate
    python run_benchmark.py --quick         # malla de puntos reducida (prueba rápida)
    python run_benchmark.py --V0 1.0 --a 0.5 --m 1.0 --E 0.5 --tol 0.05

Experimentos: validate | energy | width | height | cost | all
"""
from __future__ import annotations

import argparse
import os
import qbench
from qbench import experiments as ex
from qbench import plotting, tdse, validation
from qbench.config import Barrier


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="all", choices=["all", "validate", "energy", "width", "height", "cost"])
    ap.add_argument("--V0", type=float, default=1.0, help="altura de la barrera [eV]")
    ap.add_argument("--a", type=float, default=0.5, help="ancho de la barrera [nm]")
    ap.add_argument("--m", type=float, default=1.0, help="masa en unidades de m_e")
    ap.add_argument("--E", type=float, default=0.5, help="energía fija en los exp. 2-4 [eV]")
    ap.add_argument("--dx", type=float, default=0.02, help="dx objetivo del TDSE [nm]")
    ap.add_argument("--dt", type=float, default=0.025, help="dt del TDSE [ħ/eV]")
    ap.add_argument("--tol", type=float, default=0.05, help="tolerancia de error relativo para la ventana de validez")
    ap.add_argument("--quick", action="store_true", help="menos puntos por barrido")
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    b = Barrier(V0=args.V0, a=args.a, m=args.m)
    cfg = tdse.TDSEConfig(dx=args.dx, dt=args.dt)
    n1, n2, n3 = (12, 10, 10) if args.quick else (32, 24, 24)
    summary = []

    if args.exp in ("all", "validate"):
        print("\n[0] Validación")
        v = validation.run_validation(b, outdir=args.outdir)
        summary += v["lines"] + [""]

    sweeps = {
        "energy": lambda: ex.exp_energy(b, cfg, n1),
        "width": lambda: ex.exp_width(b, args.E, cfg, n2),
        "height": lambda: ex.exp_height(b, args.E, cfg, n3),
    }
    for i, (key, fn) in enumerate(sweeps.items(), start=1):
        if args.exp in ("all", key):
            print(f"\n[{i}] Experimento {i}")
            e = fn()
            ex.save_csv(e, args.outdir)
            plotting.plot_sweep(e, os.path.join(args.outdir, f"{i:02d}_{e['name']}.png"), tol=args.tol)
            rep = ex.report(e, args.tol)
            print(rep)
            summary += [rep, ""]

    if args.exp in ("all", "cost"):
        print("\n[4] Experimento 4")
        e = ex.exp_cost(b, args.E)
        ex.save_csv(e, args.outdir)
        plotting.plot_cost(e, os.path.join(args.outdir, f"04_{e['name']}.png"))

    if summary:
        with open(os.path.join(args.outdir, "resumen.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(summary))
    print(f"\nResultados en: {args.outdir}")


if __name__ == "__main__":
    main()
