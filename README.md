# Majorana Propagation vs. Exact Dynamics for the Fermi-Hubbard Model

Simulations of doublon dynamics in the 2D Fermi-Hubbard model, using two approaches:

1. **Exact dynamics (ED)** on small lattices (2x2, 2x4) with [NetKet](https://www.netket.org/). These serve as benchmarks.
2. **Majorana Propagation (MP)** of the doublon-number observable in the Heisenberg picture on a periodic 7x4 lattice, using [MajoranaPropagation.jl](https://github.com/SparqleSim/MajoranaPropagation.jl). The results are compared to trapped-ion data (TFLO + GPR, MP trapped ion, raw).

## Repository structure

```
.
├── ExactDiagonalization/          # Python / NetKet
│   ├── exact_dynamics/            # small package: ExactDynamics driver, state, expectation values
│   ├── exact2x2.py                # ED on a 2x2 periodic lattice
│   ├── exact2x4.py                # ED on a 2x4 periodic lattice (paired initial state)
│   ├── dynamics4.ipynb            # interactive ED notebook
│   ├── dynamics4original.ipynb    # original version of the notebook
│   └── figures/
│
└── MajoranaPropagation/           # Julia
    ├── Project.toml, Manifest.toml
    ├── Majorana7x4_newmethod.jl   # 7x4 MP simulation with penalty-based truncation
    ├── MajoranaPlot_newmethod.jl  # plotting of the results vs. trapped-ion data
    ├── submit_Maj7x4_newmethod.sh # SLURM submission script (parameter sweep)
    └── notebooks/                 # earlier exploratory simulations
```

## Exact dynamics

Requirements: `netket`, `jax`, `numpy`, `pandas`, `matplotlib`.

Run the scripts from inside `ExactDiagonalization/` so that the `exact_dynamics` package can be imported:

```bash
cd ExactDiagonalization
mkdir -p data
python exact2x4.py
```

The time evolution uses a second-order Heun integrator with `dt = 0.01` up to `T = 2`. Observables are logged to `data/` and plotted.

## Majorana Propagation

### Setup

The scripts need the `phase_hopping` branch of MajoranaPropagation.jl, which provides the phase-hopping gates:

```julia
using Pkg
Pkg.activate("MajoranaPropagation")
Pkg.add(url="https://github.com/SparqleSim/MajoranaPropagation.jl", rev="phase_hopping")
Pkg.add(["PauliPropagation", "JLD2", "ArgParse", "Combinatorics", "TimerOutputs",
         "ProgressMeter", "ColorSchemes", "Plots", "LaTeXStrings", "Measures"])
```

### Method (`Majorana7x4_newmethod.jl`)

- **Lattice:** 7x4 with periodic boundaries. The Trotter step is second order (up hoppings → down hoppings → on-site `U` → reversed), with time step `dt`.
- **Initial state:** a doublon on site 19, a hole on site 2, and 13 spin pairs `(|↑↓⟩ + |↓↑⟩)/√2` on fixed site pairs.
- **Observable:** total doublon number `Σ_i n_i↑ n_i↓`, back-propagated layer by layer. Each layer's expectation value is the overlap with the initial state.
- **Truncation:**
  - Coefficient truncation (`--min_abs_coeff`).
  - A new *penalty* truncation (`--max_penalties`). A Majorana string gets one penalty for each site pair on which it has zero overlap with the pair state, plus the number of unpaired Majoranas on the doublon/hole sites. Strings with more than `max_penalties` penalties are discarded.

### Running

```bash
cd MajoranaPropagation
julia -t 20 Majorana7x4_newmethod.jl --U 4 --n_layers 20 --dt 0.1 \
      --min_abs_coeff 1e-6 --max_penalties 4 --save_dir ./data/
julia MajoranaPlot_newmethod.jl --U 4 --n_layers 20 --save_dir ./
```

`submit_Maj7x4_newmethod.sh` submits the full parameter sweep (`max_penalties` ∈ {3, 4, 5, 6}, `min_abs_coeff = 1e-6`) to a SLURM cluster. The default `save_dir` paths point to the cluster scratch directory and need to be adapted. Runs can be resumed with `--from_checkpoint true`.

Each run stores the doublon number per layer (`n_doubl.jld2`), the number of Majorana strings (`length.jld2`), the coefficient norm (`norm.jld2`), and the serialized observable.
