# Benchmarking of Quantum-Mechanical Methods for 1D Potential Barriers

Framework en Python (numpy, scipy, sympy, matplotlib) que compara **WKB**, **TMM** y **TDSE**
sobre la misma barrera rectangular V(x)=V0 para |x|<a/2. Observable: probabilidad de transmisión T.

Unidades internas: eV, nm, ħ/eV.  Por defecto: V0 = 1 eV, a = 0.5 nm, m = mₑ.

## Estructura (un módulo por método / función)

| Módulo | Contenido |
|---|---|
| `qbench/config.py` | unidades, constante ħ²/2m, dataclass `Barrier(V0, a, m)` |
| `qbench/tmm.py` | TMM general para capas constantes a trozos + fórmula analítica (validación). **Es la referencia.** |
| `qbench/wkb.py` | WKB (Gamow y Kemble) + integral de acción numérica para potenciales genéricos |
| `qbench/tdse.py` | TDSE: split-operator (Strang + FFT), paquete gaussiano, capa absorbente, T(k) por cociente espectral |
| `qbench/symbolic.py` | derivación simbólica de t(k,κ,a) con sympy (validación independiente de TMM) |
| `qbench/metrics.py` | error relativo, error en décadas, ventanas de validez, cronometraje |
| `qbench/validation.py` | 7 chequeos automáticos antes del benchmark |
| `qbench/experiments.py` | Exp.1 T(E), Exp.2 T(a), Exp.3 T(V0), Exp.4 costo vs precisión |
| `qbench/plotting.py` | figuras |
| `run_benchmark.py` | CLI |

## Uso

```bash
python run_benchmark.py                  # validación + 4 experimentos (~3 min)
python run_benchmark.py --exp energy     # sólo T vs E
python run_benchmark.py --quick          # prueba rápida
python run_benchmark.py --V0 2 --a 0.3 --E 0.8 --tol 0.02
```

Salida en `results/`: PNG de cada experimento, CSV con todos los datos y `resumen.txt`.

## Notas metodológicas

* **Error** = |T − T_TMM| / T_TMM (y |log10(T/T_TMM)|, porque en tunelamiento T abarca décadas).
* **TDSE**: se extrae T(k0) del cociente |F[ψ_transmitida]|²/|F[ψ_inicial]|² con un ajuste local de ln T(k);
  el splitting con potencial discontinuo produce un rizado espectral de pequeña amplitud.
* El error del TDSE está dominado por **dt** (≈ dt^1.5) y crece con V0 y con la profundidad del tunelamiento.
* **WKB** con bordes abruptos no incluye el prefactor 16E(V0−E)/V0²; por eso falla por un factor O(1)
  aunque la dependencia exponencial en a y V0 sea correcta.
