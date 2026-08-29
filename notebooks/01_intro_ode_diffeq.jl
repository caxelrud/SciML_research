### A Pluto.jl notebook ###
# 0.19.40

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ fd04ba30-d0e4-41ea-9a55-f819a14e1076
md"""
# 1 · Introduction to solving ODEs with DifferentialEquations.jl

This notebook is a first tour of the SciML differential-equation solvers, from a plain exponential decay to a chaotic system, ending with an interactive damping slider.
"""

# ╔═╡ 3d89f05d-478c-41b5-ad7e-ef30248b0fc1
using OrdinaryDiffEq, Plots, PlutoUI

# ╔═╡ 32ea9353-7ea4-4ebe-b694-5b85b42ff4e1
md"""
## 1. Exponential decay

The simplest possible ODE: $\dot u = \lambda u$.
"""

# ╔═╡ f8762153-afa3-4662-b4f7-72ab471ad422
begin
	λ = -0.5
	u0_decay = 1.0
	tspan_decay = (0.0, 10.0)
	decay_prob = ODEProblem((u, p, t) -> λ * u, u0_decay, tspan_decay)
	decay_sol = solve(decay_prob, Tsit5())
end

# ╔═╡ c53fd23a-75f2-459a-bbd0-3cf22e8e75c1
plot(decay_sol; label="u(t)", xlabel="t", ylabel="u", title="Exponential decay  du/dt = λu")

# ╔═╡ b4e13cdd-704d-4ae7-b3a2-6aa20c7bca0c
md"""
## 2. A nonlinear system: the damped pendulum

$$\dot\theta = \omega, \qquad \dot\omega = -\frac{g}{L}\sin\theta - \gamma\omega$$
"""

# ╔═╡ 766e2bda-3412-4829-9c61-668249ccab8b
function pendulum!(du, u, p, t)
	θ, ω = u
	g, L, γ = p
	du[1] = ω
	du[2] = -(g / L) * sin(θ) - γ * ω
	return nothing
end

# ╔═╡ 2f4507b0-2a2d-404a-9fbb-ec1fd18e38c5
begin
	p_pend = (9.81, 1.0, 0.3)   # g, L, γ
	u0_pend = [π / 3, 0.0]       # initial angle, angular velocity
	tspan_pend = (0.0, 20.0)
	pend_prob = ODEProblem(pendulum!, u0_pend, tspan_pend, p_pend)
	pend_sol = solve(pend_prob, Tsit5())
end

# ╔═╡ a25fb212-6671-4d13-8596-caf776ce29db
plot(pend_sol; idxs=1, label="θ(t)", xlabel="t", ylabel="angle (rad)", title="Damped pendulum")

# ╔═╡ df957d13-af56-4341-86eb-c6fd198cf68e
md"""
## 3. Interactivity: drag the damping coefficient

Pluto reruns every dependent cell automatically whenever `γ_interactive` changes.
"""

# ╔═╡ 46e22e65-b73b-4ce2-aaec-d845061e757a
@bind γ_interactive Slider(0.0:0.05:2.0, default=0.3, show_value=true)

# ╔═╡ ee2b303f-6fcf-495e-84e4-58cd3a67639a
begin
	p_interactive = (9.81, 1.0, γ_interactive)
	interactive_prob = ODEProblem(pendulum!, u0_pend, tspan_pend, p_interactive)
	interactive_sol = solve(interactive_prob, Tsit5())
	plot(interactive_sol; idxs=1, label="θ(t), γ=$(γ_interactive)", xlabel="t", ylabel="angle (rad)",
		title="Damping slider demo", ylims=(-1.2, 1.2))
end

# ╔═╡ 429938e7-a24e-4a3c-a41e-8b754a570b9b
md"""
## 4. A taste of chaos: the Lorenz attractor

$$\dot x = \sigma(y-x), \qquad \dot y = x(\rho - z) - y, \qquad \dot z = xy - \beta z$$
"""

# ╔═╡ 997265b4-0f97-4927-b04e-2938a9dce1a4
function lorenz!(du, u, p, t)
	σ, ρ, β = p
	x, y, z = u
	du[1] = σ * (y - x)
	du[2] = x * (ρ - z) - y
	du[3] = x * y - β * z
	return nothing
end

# ╔═╡ c4cdd998-a3e0-4484-a3c1-37e07ddb6e0b
begin
	lorenz_p = (10.0, 28.0, 8 / 3)
	lorenz_u0 = [1.0, 0.0, 0.0]
	lorenz_tspan = (0.0, 40.0)
	lorenz_prob = ODEProblem(lorenz!, lorenz_u0, lorenz_tspan, lorenz_p)
	lorenz_sol = solve(lorenz_prob, Tsit5(); saveat=0.01)
end

# ╔═╡ 6ea9f785-9a0d-4faa-9342-1492d9e01f49
plot(lorenz_sol; idxs=(1, 2, 3), camera=(30, 30), label=false, title="Lorenz attractor", linewidth=0.8)

# ╔═╡ 7419ffe6-0c48-4d39-bfbd-66620693de7a
md"""
## Takeaways

* `ODEProblem(f, u0, tspan, p)` plus `solve(prob, alg)` is all you need for scalar equations, systems, and chaotic systems alike.
* Swapping the algorithm (`Tsit5()`, `Rodas5()`, ...) or tolerances is a one-line change — the modeling code never has to change.
* `@bind` from `PlutoUI` turns any notebook into a live, explorable simulation.

Next: [`02_neural_ode.jl`](./02_neural_ode.jl) replaces part of the right-hand side of an ODE with a neural network.
"""

# ╔═╡ Cell order:
# ╟─fd04ba30-d0e4-41ea-9a55-f819a14e1076
# ╠═3d89f05d-478c-41b5-ad7e-ef30248b0fc1
# ╟─32ea9353-7ea4-4ebe-b694-5b85b42ff4e1
# ╠═f8762153-afa3-4662-b4f7-72ab471ad422
# ╠═c53fd23a-75f2-459a-bbd0-3cf22e8e75c1
# ╟─b4e13cdd-704d-4ae7-b3a2-6aa20c7bca0c
# ╠═766e2bda-3412-4829-9c61-668249ccab8b
# ╠═2f4507b0-2a2d-404a-9fbb-ec1fd18e38c5
# ╠═a25fb212-6671-4d13-8596-caf776ce29db
# ╟─df957d13-af56-4341-86eb-c6fd198cf68e
# ╠═46e22e65-b73b-4ce2-aaec-d845061e757a
# ╠═ee2b303f-6fcf-495e-84e4-58cd3a67639a
# ╟─429938e7-a24e-4a3c-a41e-8b754a570b9b
# ╠═997265b4-0f97-4927-b04e-2938a9dce1a4
# ╠═c4cdd998-a3e0-4484-a3c1-37e07ddb6e0b
# ╠═6ea9f785-9a0d-4faa-9342-1492d9e01f49
# ╟─7419ffe6-0c48-4d39-bfbd-66620693de7a
