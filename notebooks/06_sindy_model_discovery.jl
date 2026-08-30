### A Pluto.jl notebook ###
# 0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ 263b8a59-fd27-40fb-8dba-6346faac0cb9
md"""
# 6 · Discovering governing equations from data (SINDy)

The previous notebooks all assumed a model *structure* and either fit its parameters or filled a gap with a neural network. This notebook goes further: given only a measured trajectory, can we recover the **symbolic equations of motion** themselves, with no equation supplied at all?

We use **SINDy** (Sparse Identification of Nonlinear Dynamics, Brunton, Proctor & Kutz 2016) via `DataDrivenDiffEq.jl`: build a large library of candidate nonlinear terms, then use sparse regression to keep only the handful that are actually needed. We apply it to the Lorenz system.
"""

# ╔═╡ fa04b521-0dbd-4987-a1f1-b2a122170fc1
using DataDrivenDiffEq, DataDrivenSparse, ModelingToolkit, OrdinaryDiffEq, LinearAlgebra, Random, Plots

# ╔═╡ 129ee7ed-28cd-4740-ae91-08a52d5d775c
md"""
## 1. Generate trajectory data from the (secretly known) Lorenz system

In a real application this data would come from an experiment or a simulation whose equations we don't fully trust or don't have. Here we generate it from the classic Lorenz equations so we can check whether SINDy recovers them.
"""

# ╔═╡ 08781e89-ba98-47c6-ac38-fc03449d10a9
function lorenz!(du, u, p, t)
	σ, ρ, β = p
	x, y, z = u
	du[1] = σ * (y - x)
	du[2] = x * (ρ - z) - y
	du[3] = x * y - β * z
	return nothing
end

# ╔═╡ 8b032923-b687-4a73-acbe-c080c651020d
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

# ╔═╡ 8eed2a4e-6bcb-453f-be29-6404d34cc124
plot(lorenz_sol6; idxs=(1, 2, 3), label=false, linewidth=0.7, title="Data given to SINDy (Lorenz trajectory)")

# ╔═╡ 4cc69d54-4530-45c8-ae13-46477665d9ba
md"""
## 2. Wrap the data as a `DataDrivenProblem`

`DataDrivenDiffEq.jl` can estimate the derivatives $\dot X$ from the trajectory itself (numerical differentiation / collocation), so we don't even need to supply them directly — though supplying exact derivatives (as we do here, since we can compute them) gives a cleaner result for a first example.
"""

# ╔═╡ 318ac9f4-010d-47fa-a2d6-0fbda42599ea
begin
	DX = similar(X)
	for (i, ui) in enumerate(eachcol(X))
		lorenz!(view(DX, :, i), ui, lorenz_p6, t6[i])
	end
	ddprob = ContinuousDataDrivenProblem(X, t6; DX=DX)
end

# ╔═╡ 5849f714-a3cb-4f8b-ac6c-6b8556aa0d2b
md"""
## 3. Build a candidate function library

SINDy needs a dictionary of candidate terms to choose from — here, all monomials up to degree 2 in $(x, y, z)$, which is more than enough to contain the true quadratic Lorenz right-hand side.
"""

# ╔═╡ 2313cae0-d981-4759-9b9a-81f56aff5327
begin
	@variables x y z
	state_vars = [x, y, z]
	basis = Basis(polynomial_basis(state_vars, 2), state_vars)
end

# ╔═╡ c6c95b6a-54ac-408f-a740-3ee48b8515e9
md"""
## 4. Sparse regression

We search for the sparsest set of coefficients (over the candidate library) that still reproduces $\dot X$ accurately, using the STLSQ (sequential thresholded least squares) algorithm at a range of sparsity thresholds.
"""

# ╔═╡ f1f0c5ae-0d64-475e-8936-57b200a198ff
begin
	sparse_opt = STLSQ(exp10.(-4:0.25:1))
	sindy_res = solve(ddprob, basis, sparse_opt)
	sindy_sys = get_basis(sindy_res)
end

# ╔═╡ 29528597-2d1c-462a-ab7f-0414a71d80ea
println(sindy_sys)

# ╔═╡ cb0ffe2d-fdb2-4d6b-a571-2be5b587db93
get_parameter_values(sindy_sys)

# ╔═╡ 433c2961-0558-4186-b152-4e3ad6f18cf6
md"""
## 5. Validate: simulate the discovered model and compare

The discovered symbolic model can be converted straight back into an `ODEProblem` and simulated exactly like a hand-written one — closing the loop from data, to equations, to predictions.
"""

# ╔═╡ 5e798489-0492-4754-a178-b300f95f055c
begin
	discovered_prob = ODEProblem(sindy_sys, u0_6, tspan_6, get_parameter_values(sindy_sys))
	discovered_sol = solve(discovered_prob, Tsit5(); saveat=dt6)

	plot(lorenz_sol6; idxs=1, label="true x(t)", lw=2)
	plot!(discovered_sol; idxs=1, label="SINDy-discovered x(t)", lw=2, ls=:dash)
end

# ╔═╡ 181319c9-daea-4067-a2df-231ddcd4786c
md"""
## Takeaways

* SINDy turns model discovery into a sparse regression problem: pick the fewest library terms that explain the observed derivatives.
* Because the result is a short symbolic expression rather than a black-box network, it is directly interpretable — you can read off the discovered coefficients and compare them to $\sigma, \rho, \beta$.
* This is the natural companion to the Universal Differential Equations notebook: train a neural network on the unknown part of a model, then run SINDy on the network's input/output pairs to turn it into a genuine equation.
"""

# ╔═╡ Cell order:
# ╟─263b8a59-fd27-40fb-8dba-6346faac0cb9
# ╠═fa04b521-0dbd-4987-a1f1-b2a122170fc1
# ╟─129ee7ed-28cd-4740-ae91-08a52d5d775c
# ╠═08781e89-ba98-47c6-ac38-fc03449d10a9
# ╠═8b032923-b687-4a73-acbe-c080c651020d
# ╠═8eed2a4e-6bcb-453f-be29-6404d34cc124
# ╟─4cc69d54-4530-45c8-ae13-46477665d9ba
# ╠═318ac9f4-010d-47fa-a2d6-0fbda42599ea
# ╟─5849f714-a3cb-4f8b-ac6c-6b8556aa0d2b
# ╠═2313cae0-d981-4759-9b9a-81f56aff5327
# ╟─c6c95b6a-54ac-408f-a740-3ee48b8515e9
# ╠═f1f0c5ae-0d64-475e-8936-57b200a198ff
# ╠═29528597-2d1c-462a-ab7f-0414a71d80ea
# ╠═cb0ffe2d-fdb2-4d6b-a571-2be5b587db93
# ╟─433c2961-0558-4186-b152-4e3ad6f18cf6
# ╠═5e798489-0492-4754-a178-b300f95f055c
# ╟─181319c9-daea-4067-a2df-231ddcd4786c
