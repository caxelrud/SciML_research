### A Pluto.jl notebook ###
# 0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ 86c9364b-763c-4200-b9e7-fc7b0e4ab470
md"""
# O&G · 6 — Discovering the decline law itself from production data

Every earlier notebook in this series *assumed* a model form (Arps, a Fetkovich aquifer, the diffusivity equation) and either fit its parameters or filled a gap with a neural network. This closing notebook asks a more basic question: given nothing but a rate-vs-time history, can we discover the governing decline **equation** itself, with no assumed functional form at all?

We use **SINDy** ([`06_sindy_model_discovery.jl`](../06_sindy_model_discovery.jl)) on a well whose true (but, to the algorithm, unknown) behavior is harmonic decline ($b=1$ in the Arps family), which happens to correspond to the clean autonomous ODE $\dot q = -\frac{D_i}{q_i}q^2$ — a single quadratic term. If SINDy is given a library rich enough to contain that term, sparse regression should isolate it automatically.
"""

# ╔═╡ 49a35caa-69a3-4b94-b0f2-ffa2c722cae5
using DataDrivenDiffEq, DataDrivenSparse, ModelingToolkit, OrdinaryDiffEq, Random, Plots

# ╔═╡ f88dd5a8-c09e-40e6-b45d-13bb01e1fdeb
md"""
## 1. A harmonically declining well

$$\dot q = -\frac{D_i}{q_i}\,q^2$$
"""

# ╔═╡ 34dd3c94-eae7-430a-9d0b-53a83695423c
function harmonic_decline!(du, u, p, t)
	Di, qi = p
	q = u[1]
	du[1] = -Di * q^2 / qi
	return nothing
end

# ╔═╡ 86d1e37e-f745-45a4-91fd-9c111e41c9b3
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

# ╔═╡ 51078487-a34e-4ab5-9c7f-f49d4938667c
plot(tq, Xq[1, :]; xlabel="time (yr)", ylabel="q (bbl/d)", label="production history given to SINDy")

# ╔═╡ eb3f58de-d826-4fd6-a21d-e4ee2e26d799
md"""
## 2. Build the `DataDrivenProblem`

As in the general SINDy notebook, we supply the derivative directly here (computable because we control the synthetic example); in practice `DataDrivenDiffEq.jl` can also estimate it from the trajectory by finite differencing or collocation.
"""

# ╔═╡ a8cf1ae7-ab8e-4f72-8c91-0315e533de25
begin
	DXq = similar(Xq)
	for (i, ui) in enumerate(eachcol(Xq))
		harmonic_decline!(view(DXq, :, i), ui, (Di_og6, qi_og6), tq[i])
	end
	ddprob_q = ContinuousDataDrivenProblem(Xq, tq; DX=DXq)
end

# ╔═╡ 726ea958-d8b2-498b-8b94-6ee41fbde811
md"""
## 3. A polynomial candidate library

We deliberately do *not* tell SINDy the true exponent — the library contains every monomial in $q$ up to degree 3, and sparse regression must pick out the one that actually matters.
"""

# ╔═╡ 166cd62f-1ca5-4143-9de1-07d714a7a165
begin
	@variables q
	decline_basis = Basis(polynomial_basis([q], 3), [q])
end

# ╔═╡ 124b8a9a-12c4-437f-aa39-aa04579d1954
begin
	sparse_opt_q = STLSQ(exp10.(-6:0.25:1))
	sindy_res_q = solve(ddprob_q, decline_basis, sparse_opt_q)
	sindy_decline_law = get_basis(sindy_res_q)
end

# ╔═╡ e31da516-ee31-4efa-80a7-49aa2fb5886e
println(sindy_decline_law)

# ╔═╡ a16dcacc-8b03-47d0-98e7-fd8fc13e8fb1
get_parameter_values(sindy_res_q)

# ╔═╡ fe9e040a-bf82-4163-99c8-979c9a2de256
md"""
## 4. Validate the discovered law by forecasting forward
"""

# ╔═╡ 52c47181-c317-41f6-a843-129c3afa9e79
begin
	discovered_well_prob = ODEProblem(sindy_decline_law, [qi_og6], tspan_og6, get_parameter_values(sindy_res_q))
	discovered_well_sol = solve(discovered_well_prob, Tsit5(); saveat=saveat_og6)

	plot(tq, Xq[1, :]; label="true harmonic decline", lw=2, xlabel="time (yr)", ylabel="q (bbl/d)")
	plot!(tq, Array(discovered_well_sol)[1, :]; label="SINDy-discovered model", lw=2, ls=:dash)
end

# ╔═╡ bd775f82-8f23-4d73-9ce9-d31f64be0596
md"""
## Takeaways

* Given a rich-enough candidate library, SINDy recovered that this well's decline is governed by a single $q^2$ term — i.e. it independently rediscovered harmonic ($b=1$) Arps decline directly from the rate history, without that model form ever being stated.
* A non-integer decline exponent ($0<b<1$, the common hyperbolic case) is not a single monomial, so a plain polynomial library would only approximate it; a production application would extend the library with fractional powers of $q$ or apply SINDy locally over the operating range, but the mechanics — build a library, run sparse regression, validate by simulating forward — stay identical.
* This closes the loop on the notebook series: [`01_decline_curve_analysis.jl`](./01_decline_curve_analysis.jl) assumed the Arps form and fit its parameters; this notebook shows that, at least for the clean cases, the form itself doesn't have to be assumed either.
"""

# ╔═╡ Cell order:
# ╟─86c9364b-763c-4200-b9e7-fc7b0e4ab470
# ╠═49a35caa-69a3-4b94-b0f2-ffa2c722cae5
# ╟─f88dd5a8-c09e-40e6-b45d-13bb01e1fdeb
# ╠═34dd3c94-eae7-430a-9d0b-53a83695423c
# ╠═86d1e37e-f745-45a4-91fd-9c111e41c9b3
# ╠═51078487-a34e-4ab5-9c7f-f49d4938667c
# ╟─eb3f58de-d826-4fd6-a21d-e4ee2e26d799
# ╠═a8cf1ae7-ab8e-4f72-8c91-0315e533de25
# ╟─726ea958-d8b2-498b-8b94-6ee41fbde811
# ╠═166cd62f-1ca5-4143-9de1-07d714a7a165
# ╠═124b8a9a-12c4-437f-aa39-aa04579d1954
# ╠═e31da516-ee31-4efa-80a7-49aa2fb5886e
# ╠═a16dcacc-8b03-47d0-98e7-fd8fc13e8fb1
# ╟─fe9e040a-bf82-4163-99c8-979c9a2de256
# ╠═52c47181-c317-41f6-a843-129c3afa9e79
# ╟─bd775f82-8f23-4d73-9ce9-d31f64be0596
