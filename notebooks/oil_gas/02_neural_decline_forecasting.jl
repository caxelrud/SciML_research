### A Pluto.jl notebook ###
# 0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ fa34b36c-16e3-475c-9e57-3d9ab843b60e
md"""
# O&G · 2 — Neural ODEs for production that doesn't follow Arps

Arps decline curves ([`01_decline_curve_analysis.jl`](./01_decline_curve_analysis.jl)) assume smooth, boundary-dominated depletion. Real wells often deviate: workovers, temporary shut-ins, choke changes, and gauge-cycling all leave signatures a simple decline law can't represent. A **neural ODE** $\dot q = f_\theta(q, t)$ makes no such assumption — it learns whatever dynamics the data actually show.
"""

# ╔═╡ a4a01c27-10e6-4431-b3a4-50909648fd49
using Lux, OrdinaryDiffEq, SciMLSensitivity, Optimization, OptimizationOptimisers, ComponentArrays, Random, Plots

# ╔═╡ 99f6d4e2-c279-4eee-b588-225b40a5a242
md"""
## 1. A well with an irregular production history

We generate synthetic data from an *underlying* hyperbolic decline that is disturbed by a temporary shut-in (a dip around 1.5 years, e.g. for a workover) and a small periodic ripple (choke cycling). In a real dataset we would not know this structure — we only see the noisy rate history.
"""

# ╔═╡ 4cab93f6-78b2-4813-baaf-fda0a30f1036
function true_well_dynamics!(du, u, p, t)
	q = max(u[1], 0.0)   # production rate can't go negative; guards the fractional power below
	qi, Di, b = p
	arps_term = -Di * q^(1 + b) / qi^b
	workover_dip = -0.6 * q * exp(-((t - 1.5)^2) / (2 * 0.06^2))
	choke_ripple = 0.08 * q * sin(2π * t / 0.25)
	du[1] = arps_term + workover_dip + choke_ripple
	return nothing
end

# ╔═╡ d3e446ab-57b4-4daf-a7c8-53b1dba9f712
begin
	rng_og2 = Random.default_rng()
	Random.seed!(rng_og2, 5)

	true_p_og2 = (1000.0, 0.8, 0.5)   # qi, Di, b
	u0_og2 = [1000.0]
	tspan_og2 = (0.0, 3.0)
	tsteps_og2 = range(tspan_og2[1], tspan_og2[2]; length=60)

	true_prob_og2 = ODEProblem(true_well_dynamics!, u0_og2, tspan_og2, true_p_og2)
	true_sol_og2 = solve(true_prob_og2, Tsit5(); saveat=tsteps_og2)
	well_data = Array(true_sol_og2) .* (1 .+ 0.03 .* randn(rng_og2, 1, length(tsteps_og2)))
end

# ╔═╡ e9d92f23-3096-4032-b245-d4ee211a82a8
scatter(tsteps_og2, well_data[1, :]; label="reported rate", xlabel="time (yr)", ylabel="q (bbl/d)",
	title="Irregular production history (workover dip + choke ripple)")

# ╔═╡ 8ca48dcd-d1f6-4e76-a0ca-1acff0d298bb
md"""
## 2. A neural ODE with no assumed decline law

The network sees both the current rate and the time, so it can in principle represent time-driven effects like the workover and the periodic ripple, not just state-driven decline.
"""

# ╔═╡ 18bac39f-1a1f-4bea-bece-5c52db48d55f
begin
	qnet = Chain(Dense(2, 20, tanh), Dense(20, 20, tanh), Dense(20, 1))
	qnet_ps, qnet_st = Lux.setup(rng_og2, qnet)
	qnet_ps = ComponentArray(qnet_ps)
end

# ╔═╡ 6ed05fb4-e391-460a-9b29-98f34d8a8f5b
function neural_well_dynamics(u, p, t)
	input = [u[1] / 1000.0, t]   # rescale the rate for a well-conditioned network input
	ŷ, _ = qnet(input, p, qnet_st)
	return 1000.0 .* ŷ
end

# ╔═╡ 5bbabdba-bc04-446e-b60b-dd304031ea95
neural_well_prob = ODEProblem(neural_well_dynamics, u0_og2, tspan_og2)

# ╔═╡ ec99cbea-e357-4307-b3eb-7903e36fcd29
md"""
## 3. Train against the noisy history
"""

# ╔═╡ 0b62623d-ef6a-496a-8209-4ff1e99102e2
function predict_well(p)
	solve(neural_well_prob, Tsit5(); p=p, saveat=tsteps_og2,
		sensealg=InterpolatingAdjoint(; autojacvec=ZygoteVJP())) |> Array
end

# ╔═╡ 7b4bdab5-d399-48f3-b4d5-d165aaec9cf7
function loss_well(p, _)
	pred = predict_well(p)
	size(pred) == size(well_data) || return Inf
	return sum(abs2, well_data .- pred)
end

# ╔═╡ 38724ff9-5eb0-4e49-adca-73da63fd312a
begin
	optf_og2 = OptimizationFunction(loss_well, Optimization.AutoZygote())
	optprob_og2 = OptimizationProblem(optf_og2, qnet_ps)

	well_losses = Float64[]
	well_callback = function (state, loss)
		push!(well_losses, loss)
		return false
	end

	res_well = Optimization.solve(optprob_og2, OptimizationOptimisers.Adam(0.02); callback=well_callback, maxiters=600)
end

# ╔═╡ 9aaa67ac-502d-45d7-b5d7-2dafff0733bc
plot(well_losses; xlabel="iteration", ylabel="loss", yscale=:log10, label="training loss")

# ╔═╡ 8f72054a-a4e1-4aef-8db1-f832ba3c6fa1
md"""
## 4. Does the neural ODE capture the workover and the ripple?
"""

# ╔═╡ 0012d2b1-e730-4d93-9d7a-dd9cbf1cbb47
begin
	neural_fit = predict_well(res_well.u)
	scatter(tsteps_og2, well_data[1, :]; label="data", xlabel="time (yr)", ylabel="q (bbl/d)")
	plot!(tsteps_og2, neural_fit[1, :]; label="neural ODE fit", lw=2)
end

# ╔═╡ d55da8a5-cb44-4d5b-9611-10c2e4a09410
md"""
## Takeaways

* When the true production mechanism includes operational disturbances that don't belong in any textbook decline equation, a neural ODE fits them anyway — at the cost of a fully black-box model with no physical parameters to report.
* Feeding both $q$ and $t$ into the network lets it represent explicitly time-driven effects (the workover, the periodic ripple), not just state feedback.
* This flexibility-versus-interpretability trade-off motivates the next notebook: keep the *known* physics as an equation, and let a neural network absorb only the part you can't otherwise model.
"""

# ╔═╡ Cell order:
# ╟─fa34b36c-16e3-475c-9e57-3d9ab843b60e
# ╠═a4a01c27-10e6-4431-b3a4-50909648fd49
# ╟─99f6d4e2-c279-4eee-b588-225b40a5a242
# ╠═4cab93f6-78b2-4813-baaf-fda0a30f1036
# ╠═d3e446ab-57b4-4daf-a7c8-53b1dba9f712
# ╠═e9d92f23-3096-4032-b245-d4ee211a82a8
# ╟─8ca48dcd-d1f6-4e76-a0ca-1acff0d298bb
# ╠═18bac39f-1a1f-4bea-bece-5c52db48d55f
# ╠═6ed05fb4-e391-460a-9b29-98f34d8a8f5b
# ╠═5bbabdba-bc04-446e-b60b-dd304031ea95
# ╟─ec99cbea-e357-4307-b3eb-7903e36fcd29
# ╠═0b62623d-ef6a-496a-8209-4ff1e99102e2
# ╠═7b4bdab5-d399-48f3-b4d5-d165aaec9cf7
# ╠═38724ff9-5eb0-4e49-adca-73da63fd312a
# ╠═9aaa67ac-502d-45d7-b5d7-2dafff0733bc
# ╟─8f72054a-a4e1-4aef-8db1-f832ba3c6fa1
# ╠═0012d2b1-e730-4d93-9d7a-dd9cbf1cbb47
# ╟─d55da8a5-cb44-4d5b-9611-10c2e4a09410
