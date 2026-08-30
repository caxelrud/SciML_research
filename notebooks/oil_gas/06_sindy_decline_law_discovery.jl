### A Pluto.jl notebook ###
# 0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ 212b5cc1-7b88-4c8d-94c9-d96d84dcbb2c
md"""
# O&G · 6 — Discovering the decline law itself from production data

Every earlier notebook in this series *assumed* a model form (Arps, a Fetkovich aquifer, the diffusivity equation) and either fit its parameters or filled a gap with a neural network. This closing notebook asks a more basic question: given nothing but a rate-vs-time history, can we discover the governing decline **equation** itself, with no assumed functional form at all?

We use **SINDy** ([`06_sindy_model_discovery.jl`](../06_sindy_model_discovery.jl)) on a well whose true (but, to the algorithm, unknown) behavior is harmonic decline ($b=1$ in the Arps family), which happens to correspond to the clean autonomous ODE $\dot q = -\frac{D_i}{q_i}q^2$ — a single quadratic term. If SINDy is given a library rich enough to contain that term, sparse regression should isolate it automatically.
"""

# ╔═╡ 4db4239c-63e1-4bf0-9379-52d44faed166
using DataDrivenDiffEq, DataDrivenSparse, ModelingToolkit, OrdinaryDiffEq, Random, Plots

# ╔═╡ a9fd94b6-bdf6-4107-bd8b-52e6f622c62e
md"""
## 1. A harmonically declining well

$$\dot q = -\frac{D_i}{q_i}\,q^2$$
"""

# ╔═╡ f8ffbffb-328a-4216-9f8e-dacdbf7c619b
function harmonic_decline!(du, u, p, t)
	Di, qi = p
	q = u[1]
	du[1] = -Di * q^2 / qi
	return nothing
end

# ╔═╡ 827b1a66-7e38-486c-be07-7277ab43cdf6
begin
	rng_og6 = Random.default_rng()
	Random.seed!(rng_og6, 9)

	qi_og6, Di_og6 = 900.0, 0.9   # bbl/d, 1/yr
	tspan_og6 = (0.0, 6.0)
	saveat_og6 = 0.01

	well_prob = ODEProblem(harmonic_decline!, [qi_og6], tspan_og6, (Di_og6, qi_og6))
	well_sol = solve(well_prob, Tsit5(); saveat=saveat_og6)

	Xq = Array(well_sol)
	tq = well_sol.t
end

# ╔═╡ 59865103-1f04-4833-baa5-22b77d8c9ea5
plot(tq, Xq[1, :]; xlabel="time (yr)", ylabel="q (bbl/d)", label="production history given to SINDy")

# ╔═╡ 2b1f3603-9cb0-42f7-a45c-fbdc9d33459e
md"""
## 2. Build the `DataDrivenProblem`

As in the general SINDy notebook, we supply the derivative directly here (computable because we control the synthetic example); in practice `DataDrivenDiffEq.jl` can also estimate it from the trajectory by finite differencing or collocation.
"""

# ╔═╡ df7ee3e5-9aa4-4314-a4f7-cb9b3e5193e4
begin
	DXq = similar(Xq)
	for (i, ui) in enumerate(eachcol(Xq))
		harmonic_decline!(view(DXq, :, i), ui, (Di_og6, qi_og6), tq[i])
	end
	ddprob_q = ContinuousDataDrivenProblem(Xq, tq; DX=DXq)
end

# ╔═╡ 92e96708-5a26-4ca0-bbb8-9a3fb4fb5ee3
md"""
## 3. A polynomial candidate library

We deliberately do *not* tell SINDy the true exponent — the library contains every monomial in $q$ up to degree 3, and sparse regression must pick out the one that actually matters.
"""

# ╔═╡ 05ce4007-c224-4966-ac4b-66dcc603322f
begin
	@variables q
	decline_basis = Basis(polynomial_basis([q], 3), [q])
end

# ╔═╡ 70b2de75-ba67-47aa-a473-191cda8c2b37
begin
	sparse_opt_q = STLSQ(exp10.(-6:0.25:1))
	sindy_res_q = solve(ddprob_q, decline_basis, sparse_opt_q)
	sindy_decline_law = get_basis(sindy_res_q)
end

# ╔═╡ 1c51cf72-4461-481c-873a-5a98a1f6309a
println(sindy_decline_law)

# ╔═╡ a422d295-5075-4c9e-bc07-f615bcfd2df4
get_parameter_values(sindy_decline_law)

# ╔═╡ 6c37245b-dc3b-44f7-9065-c3feb4c59d7a
md"""
## 4. Validate the discovered law by forecasting forward
"""

# ╔═╡ b5af3064-bf14-4d3d-a2fc-4e2f6fa4c183
begin
	discovered_well_prob = ODEProblem(sindy_decline_law, [qi_og6], tspan_og6, get_parameter_values(sindy_decline_law))
	discovered_well_sol = solve(discovered_well_prob, Tsit5(); saveat=saveat_og6)

	plot(tq, Xq[1, :]; label="true harmonic decline", lw=2, xlabel="time (yr)", ylabel="q (bbl/d)")
	plot!(tq, Array(discovered_well_sol)[1, :]; label="SINDy-discovered model", lw=2, ls=:dash)
end

# ╔═╡ d760e957-2afd-44f1-8528-faa407c0c84d
md"""
## Takeaways

* Given a rich-enough candidate library, SINDy recovered that this well's decline is governed by a single $q^2$ term — i.e. it independently rediscovered harmonic ($b=1$) Arps decline directly from the rate history, without that model form ever being stated.
* A non-integer decline exponent ($0<b<1$, the common hyperbolic case) is not a single monomial, so a plain polynomial library would only approximate it; a production application would extend the library with fractional powers of $q$ or apply SINDy locally over the operating range, but the mechanics — build a library, run sparse regression, validate by simulating forward — stay identical.
* This closes the loop on the notebook series: [`01_decline_curve_analysis.jl`](./01_decline_curve_analysis.jl) assumed the Arps form and fit its parameters; this notebook shows that, at least for the clean cases, the form itself doesn't have to be assumed either.
"""

# ╔═╡ Cell order:
# ╟─212b5cc1-7b88-4c8d-94c9-d96d84dcbb2c
# ╠═4db4239c-63e1-4bf0-9379-52d44faed166
# ╟─a9fd94b6-bdf6-4107-bd8b-52e6f622c62e
# ╠═f8ffbffb-328a-4216-9f8e-dacdbf7c619b
# ╠═827b1a66-7e38-486c-be07-7277ab43cdf6
# ╠═59865103-1f04-4833-baa5-22b77d8c9ea5
# ╟─2b1f3603-9cb0-42f7-a45c-fbdc9d33459e
# ╠═df7ee3e5-9aa4-4314-a4f7-cb9b3e5193e4
# ╟─92e96708-5a26-4ca0-bbb8-9a3fb4fb5ee3
# ╠═05ce4007-c224-4966-ac4b-66dcc603322f
# ╠═70b2de75-ba67-47aa-a473-191cda8c2b37
# ╠═1c51cf72-4461-481c-873a-5a98a1f6309a
# ╠═a422d295-5075-4c9e-bc07-f615bcfd2df4
# ╟─6c37245b-dc3b-44f7-9065-c3feb4c59d7a
# ╠═b5af3064-bf14-4d3d-a2fc-4e2f6fa4c183
# ╟─d760e957-2afd-44f1-8528-faa407c0c84d
