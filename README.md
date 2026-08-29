# SciML Research — Julia + Pluto Examples

Hands-on examples exploring the [SciML](https://sciml.ai/) ecosystem
(Scientific Machine Learning) using [Julia](https://julialang.org/) and
[Pluto.jl](https://plutojl.org/) reactive notebooks.

SciML combines classical numerical differential-equation solvers with
modern machine learning — neural networks embedded inside ODEs, automatic
discovery of governing equations from data, physics-informed neural
networks, and more.

## Notebooks

| # | Notebook | Topic | Key packages |
|---|----------|-------|---------------|
| 1 | [`01_intro_ode_diffeq.jl`](notebooks/01_intro_ode_diffeq.jl) | Solving ODEs: exponential decay, damped pendulum, Lorenz attractor | `OrdinaryDiffEq`, `Plots` |
| 2 | [`02_neural_ode.jl`](notebooks/02_neural_ode.jl) | Neural ODEs: learning a spiral's dynamics with a neural network inside the ODE right-hand side | `Lux`, `SciMLSensitivity`, `Optimization` |
| 3 | [`03_universal_differential_equations.jl`](notebooks/03_universal_differential_equations.jl) | Universal Differential Equations (UDE): mixing known physics with a learned neural correction term on Lotka–Volterra | `Lux`, `SciMLSensitivity`, `Optimization` |
| 4 | [`04_parameter_estimation.jl`](notebooks/04_parameter_estimation.jl) | Inverse problems: estimating unknown ODE parameters from noisy observations | `OrdinaryDiffEq`, `Optimization`, `SciMLSensitivity` |
| 5 | [`05_physics_informed_neural_networks.jl`](notebooks/05_physics_informed_neural_networks.jl) | Physics-Informed Neural Networks (PINNs): solving a 1D Poisson PDE without a mesh | `NeuralPDE`, `Lux`, `Optimization` |
| 6 | [`06_sindy_model_discovery.jl`](notebooks/06_sindy_model_discovery.jl) | Sparse model discovery (SINDy): recovering the Lorenz equations from trajectory data | `DataDrivenDiffEq`, `OrdinaryDiffEq` |

Each notebook is self-contained: it declares its own dependencies in a
`using`/`import` cell, and Pluto's built-in package manager resolves and
installs an isolated environment for that notebook the first time you open
it (no manual `Pkg.add` needed, but the first run does need internet
access and will take a while to precompile).

## Running the notebooks

1. Install Julia (recommended via [juliaup](https://github.com/JuliaLang/juliaup)):

   ```bash
   curl -fsSL https://install.julialang.org | sh
   ```

2. Install Pluto:

   ```bash
   julia -e 'import Pkg; Pkg.add("Pluto")'
   ```

3. Launch Pluto and open a notebook:

   ```bash
   julia -e 'import Pluto; Pluto.run()'
   ```

   Then open the desired file from `notebooks/` in the browser tab that
   appears. The first cell run per notebook will download and precompile
   its packages — this is one-time and can take several minutes,
   especially for the PINN and neural ODE notebooks.

## Suggested order

The notebooks are numbered to build on each other conceptually:
start with plain ODE solving (1), then see how a neural network can
replace or augment part of a differential equation (2–3), how to fit
model parameters to data (4), how PDEs can be solved without a mesh via
neural networks (5), and finally how to go the other direction — discover
symbolic governing equations directly from data (6).

## References

- [SciML documentation](https://docs.sciml.ai/)
- [DifferentialEquations.jl](https://docs.sciml.ai/DiffEqDocs/stable/)
- [SciMLSensitivity.jl](https://docs.sciml.ai/SciMLSensitivity/stable/) (neural/universal differential equations)
- [NeuralPDE.jl](https://docs.sciml.ai/NeuralPDE/stable/) (physics-informed neural networks)
- [DataDrivenDiffEq.jl](https://docs.sciml.ai/DataDrivenDiffEq/stable/) (SINDy / sparse model discovery)
- [Optimization.jl](https://docs.sciml.ai/Optimization/stable/)
- [Pluto.jl](https://plutojl.org/)
