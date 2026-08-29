### A Pluto.jl notebook ###
# 0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ a0f245a8-928d-4ecf-8f05-7b24d0d30f40
md"""
# O&G · 2 — Neural ODEs for production that doesn't follow Arps

Arps decline curves ([`01_decline_curve_analysis.jl`](./01_decline_curve_analysis.jl)) assume smooth, boundary-dominated depletion. Real wells often deviate: workovers, temporary shut-ins, choke changes, and gauge-cycling all leave signatures a simple decline law can't represent. A **neural ODE** $\dot q = f_\theta(q, t)$ makes no such assumption — it learns whatever dynamics the data actually show.
"""

# ╔═╡ ed76cfd8-afe7-4733-b8b5-16d9775409d6
using Lux, OrdinaryDiffEq, SciMLSensitivity, Optimization, OptimizationOptimisers, ComponentArrays, Random, Plots

# ╔═╡ a22f41f4-ab9e-4b50-99a5-f575387f0fb9
md"""
## 1. A well with an irregular production history

We generate synthetic data from an *underlying* hyperbolic decline that is disturbed by a temporary shut-in (a dip around 1.5 years, e.g. for a workover) and a small periodic ripple (choke cycling). In a real dataset we would not know this structure — we only see the noisy rate history.
"""

# ╔═╡ dd69a1dc-17f6-455f-b553-61576ce1c683
function true_well_dynamics!(du, u, p, t)
	q = u[1]
	qi, Di, b = p
	arps_term = -Di * q^(1 + b) / qi^b
	workover_dip = -0.9 * q * exp(-((t - 1.5)^2) / (2 * 0.03^2))
	choke_ripple = 0.08 * q * sin(2π * t / 0.25)
	du[1] = arps_term + workover_dip + choke_ripple
	return nothing
end

# ╔═╡ 2fc0fe76-0478-4d2c-aa5f-a23d414ea384
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

# ╔═╡ e03042df-8f19-4d79-b329-926d92ba129d
scatter(tsteps_og2, well_data[1, :]; label="reported rate", xlabel="time (yr)", ylabel="q (bbl/d)",
	title="Irregular production history (workover dip + choke ripple)")

# ╔═╡ 25fbcfc1-0ae1-4859-b7e2-5fbbe24cab16
md"""
## 2. A neural ODE with no assumed decline law

The network sees both the current rate and the time, so it can in principle represent time-driven effects like the workover and the periodic ripple, not just state-driven decline.
"""

# ╔═╡ 813a5337-0637-44b5-915b-5e28f2c7c693
begin
	qnet = Chain(Dense(2, 20, tanh), Dense(20, 20, tanh), Dense(20, 1))
	qnet_ps, qnet_st = Lux.setup(rng_og2, qnet)
	qnet_ps = ComponentArray(qnet_ps)
end

# ╔═╡ fea1bfcb-6fa0-4cde-8e0f-4834a67a38ad
function neural_well_dynamics(u, p, t)
	input = [u[1] / 1000.0, t]   # rescale the rate for a well-conditioned network input
	ŷ, _ = qnet(input, p, qnet_st)
	return 1000.0 .* ŷ
end

# ╔═╡ f26e9ce9-2e93-4cfe-aae7-30789e791e2b
neural_well_prob = ODEProblem(neural_well_dynamics, u0_og2, tspan_og2)

# ╔═╡ 31d561dd-e507-467d-ac74-6ac3a4d13bf3
md"""
## 3. Train against the noisy history
"""

# ╔═╡ b1c55615-3082-48ff-91d2-ea9f12890a64
function predict_well(p)
	solve(neural_well_prob, Tsit5(); p=p, saveat=tsteps_og2,
		sensealg=InterpolatingAdjoint(; autojacvec=ZygoteVJP())) |> Array
end

# ╔═╡ 04ca8edc-8be8-4fef-846b-09c24ecbfe46
function loss_well(p, _)
	pred = predict_well(p)
	size(pred) == size(well_data) || return Inf
	return sum(abs2, well_data .- pred)
end

# ╔═╡ 0c304947-e242-48aa-9bdd-8fc79e0e6a5d
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

# ╔═╡ 2af907f8-e14f-4b48-bba3-712380bc59bd
plot(well_losses; xlabel="iteration", ylabel="loss", yscale=:log10, label="training loss")

# ╔═╡ d773e6de-1a74-46ca-b3a8-4c575adb5842
md"""
## 4. Does the neural ODE capture the workover and the ripple?
"""

# ╔═╡ 175ed34a-ba0d-45d5-849a-f988695e4e95
begin
	neural_fit = predict_well(res_well.u)
	scatter(tsteps_og2, well_data[1, :]; label="data", xlabel="time (yr)", ylabel="q (bbl/d)")
	plot!(tsteps_og2, neural_fit[1, :]; label="neural ODE fit", lw=2)
end

# ╔═╡ dad660e9-b29a-4170-ae18-0be426f99772
md"""
## Takeaways

* When the true production mechanism includes operational disturbances that don't belong in any textbook decline equation, a neural ODE fits them anyway — at the cost of a fully black-box model with no physical parameters to report.
* Feeding both $q$ and $t$ into the network lets it represent explicitly time-driven effects (the workover, the periodic ripple), not just state feedback.
* This flexibility-versus-interpretability trade-off motivates the next notebook: keep the *known* physics as an equation, and let a neural network absorb only the part you can't otherwise model.
"""

# ╔═╡ Cell order:
# ╟─a0f245a8-928d-4ecf-8f05-7b24d0d30f40
# ╠═ed76cfd8-afe7-4733-b8b5-16d9775409d6
# ╟─a22f41f4-ab9e-4b50-99a5-f575387f0fb9
# ╠═dd69a1dc-17f6-455f-b553-61576ce1c683
# ╠═2fc0fe76-0478-4d2c-aa5f-a23d414ea384
# ╠═e03042df-8f19-4d79-b329-926d92ba129d
# ╟─25fbcfc1-0ae1-4859-b7e2-5fbbe24cab16
# ╠═813a5337-0637-44b5-915b-5e28f2c7c693
# ╠═fea1bfcb-6fa0-4cde-8e0f-4834a67a38ad
# ╠═f26e9ce9-2e93-4cfe-aae7-30789e791e2b
# ╟─31d561dd-e507-467d-ac74-6ac3a4d13bf3
# ╠═b1c55615-3082-48ff-91d2-ea9f12890a64
# ╠═04ca8edc-8be8-4fef-846b-09c24ecbfe46
# ╠═0c304947-e242-48aa-9bdd-8fc79e0e6a5d
# ╠═2af907f8-e14f-4b48-bba3-712380bc59bd
# ╟─d773e6de-1a74-46ca-b3a8-4c575adb5842
# ╠═175ed34a-ba0d-45d5-849a-f988695e4e95
# ╟─dad660e9-b29a-4170-ae18-0be426f99772
