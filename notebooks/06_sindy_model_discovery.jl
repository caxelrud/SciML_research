### A Pluto.jl notebook ###
# 0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ 91f4246b-9c5c-4859-88dc-c50a7ab448b7
md"""
# 6 · Discovering governing equations from data (SINDy)

The previous notebooks all assumed a model *structure* and either fit its parameters or filled a gap with a neural network. This notebook goes further: given only a measured trajectory, can we recover the **symbolic equations of motion** themselves, with no equation supplied at all?

We use **SINDy** (Sparse Identification of Nonlinear Dynamics, Brunton, Proctor & Kutz 2016) via `DataDrivenDiffEq.jl`: build a large library of candidate nonlinear terms, then use sparse regression to keep only the handful that are actually needed. We apply it to the Lorenz system.
"""

# ╔═╡ 0f8fe50c-ee61-4823-b8e6-6c6b9ea2003a
using DataDrivenDiffEq, DataDrivenSparse, ModelingToolkit, OrdinaryDiffEq, LinearAlgebra, Random, Plots

# ╔═╡ 55334a05-c253-40c9-998f-b0d0d4049d8e
md"""
## 1. Generate trajectory data from the (secretly known) Lorenz system

In a real application this data would come from an experiment or a simulation whose equations we don't fully trust or don't have. Here we generate it from the classic Lorenz equations so we can check whether SINDy recovers them.
"""

# ╔═╡ b72f672e-c62d-4ada-ba3a-61951383e649
function lorenz!(du, u, p, t)
	σ, ρ, β = p
	x, y, z = u
	du[1] = σ * (y - x)
	du[2] = x * (ρ - z) - y
	du[3] = x * y - β * z
	return nothing
end

# ╔═╡ 8803899d-91d7-40bc-807c-b4009ecaa34b
begin
	rng6 = Random.default_rng()
	Random.seed!(rng6, 11)

	lorenz_p6 = (10.0, 28.0, 8 / 3)
	u0_6 = [1.0, 0.0, 0.0]
	tspan_6 = (0.0, 10.0)
	dt6 = 0.002

	lorenz_prob6 = ODEProblem(lorenz!, u0_6, tspan_6, lorenz_p6)
	lorenz_sol6 = solve(lorenz_prob6, Tsit5(); saveat=dt6)

	X = Array(lorenz_sol6)                    # 3 × N state trajectory
	t6 = lorenz_sol6.t
end

# ╔═╡ 08f7ac52-50e5-48bc-aa59-d58118caa1f2
plot(lorenz_sol6; idxs=(1, 2, 3), label=false, linewidth=0.7, title="Data given to SINDy (Lorenz trajectory)")

# ╔═╡ 18d0e2bd-1c19-4997-8f6c-47f066db1ddc
md"""
## 2. Wrap the data as a `DataDrivenProblem`

`DataDrivenDiffEq.jl` can estimate the derivatives $\dot X$ from the trajectory itself (numerical differentiation / collocation), so we don't even need to supply them directly — though supplying exact derivatives (as we do here, since we can compute them) gives a cleaner result for a first example.
"""

# ╔═╡ c605e0c5-6d3c-471e-ad74-c825e5a2e404
begin
	DX = similar(X)
	for (i, ui) in enumerate(eachcol(X))
		lorenz!(view(DX, :, i), ui, lorenz_p6, t6[i])
	end
	ddprob = ContinuousDataDrivenProblem(X, t6; DX=DX)
end

# ╔═╡ 56b3e635-abce-478b-b358-6321bc401671
md"""
## 3. Build a candidate function library

SINDy needs a dictionary of candidate terms to choose from — here, all monomials up to degree 2 in $(x, y, z)$, which is more than enough to contain the true quadratic Lorenz right-hand side.
"""

# ╔═╡ 11e32f26-e734-4084-a0b6-e7f876caf15b
begin
	@variables x y z
	state_vars = [x, y, z]
	basis = Basis(polynomial_basis(state_vars, 2), state_vars)
end

# ╔═╡ e065c9b9-c8e8-454f-ba94-91acc1ce2290
md"""
## 4. Sparse regression

We search for the sparsest set of coefficients (over the candidate library) that still reproduces $\dot X$ accurately, using the STLSQ (sequential thresholded least squares) algorithm at a range of sparsity thresholds.
"""

# ╔═╡ d89757ec-0ac3-4275-997e-5b276f5f4917
begin
	sparse_opt = STLSQ(exp10.(-4:0.25:1))
	sindy_res = solve(ddprob, basis, sparse_opt)
	sindy_sys = get_basis(sindy_res)
end

# ╔═╡ e9efec3c-faf3-487f-a6a4-acbb50f81ac0
println(sindy_sys)

# ╔═╡ b17ef7c3-5540-4e23-8004-450961cfd462
get_parameter_values(sindy_res)

# ╔═╡ 783f48ce-b877-4de6-abb5-7f6e52d46ec3
md"""
## 5. Validate: simulate the discovered model and compare

The discovered symbolic model can be converted straight back into an `ODEProblem` and simulated exactly like a hand-written one — closing the loop from data, to equations, to predictions.
"""

# ╔═╡ 115b5994-11ff-4b54-965f-9005921552f8
begin
	discovered_prob = ODEProblem(sindy_sys, u0_6, tspan_6, get_parameter_values(sindy_res))
	discovered_sol = solve(discovered_prob, Tsit5(); saveat=dt6)

	plot(lorenz_sol6; idxs=1, label="true x(t)", lw=2)
	plot!(discovered_sol; idxs=1, label="SINDy-discovered x(t)", lw=2, ls=:dash)
end

# ╔═╡ 72ac9998-6228-4073-a5e9-23d278fc986a
md"""
## Takeaways

* SINDy turns model discovery into a sparse regression problem: pick the fewest library terms that explain the observed derivatives.
* Because the result is a short symbolic expression rather than a black-box network, it is directly interpretable — you can read off the discovered coefficients and compare them to $\sigma, \rho, \beta$.
* This is the natural companion to the Universal Differential Equations notebook: train a neural network on the unknown part of a model, then run SINDy on the network's input/output pairs to turn it into a genuine equation.
"""

# ╔═╡ Cell order:
# ╟─91f4246b-9c5c-4859-88dc-c50a7ab448b7
# ╠═0f8fe50c-ee61-4823-b8e6-6c6b9ea2003a
# ╟─55334a05-c253-40c9-998f-b0d0d4049d8e
# ╠═b72f672e-c62d-4ada-ba3a-61951383e649
# ╠═8803899d-91d7-40bc-807c-b4009ecaa34b
# ╠═08f7ac52-50e5-48bc-aa59-d58118caa1f2
# ╟─18d0e2bd-1c19-4997-8f6c-47f066db1ddc
# ╠═c605e0c5-6d3c-471e-ad74-c825e5a2e404
# ╟─56b3e635-abce-478b-b358-6321bc401671
# ╠═11e32f26-e734-4084-a0b6-e7f876caf15b
# ╟─e065c9b9-c8e8-454f-ba94-91acc1ce2290
# ╠═d89757ec-0ac3-4275-997e-5b276f5f4917
# ╠═e9efec3c-faf3-487f-a6a4-acbb50f81ac0
# ╠═b17ef7c3-5540-4e23-8004-450961cfd462
# ╟─783f48ce-b877-4de6-abb5-7f6e52d46ec3
# ╠═115b5994-11ff-4b54-965f-9005921552f8
# ╟─72ac9998-6228-4073-a5e9-23d278fc986a
