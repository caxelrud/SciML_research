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

# ╔═╡ 4575c6e8-8ec1-4dd1-8f13-a208e28b334f
md"""
# O&G · 1 — Decline curve analysis with Arps equations

The most common forecasting tool in production engineering is the **Arps decline curve**. Rather than treat exponential, harmonic, and hyperbolic decline as three separate formulas, we write them as a single ODE and let the *decline exponent* $b$ interpolate between them — exactly the kind of unification SciML's solver stack makes easy.

The nominal decline rate is defined as $D(t) = -\frac{1}{q}\frac{dq}{dt}$. Arps assumes $D(t) = \dfrac{D_i}{1 + bD_it}$, which is equivalent to the autonomous ODE

$$\frac{dq}{dt} = -D_i\, \frac{q^{1+b}}{q_i^{\,b}}, \qquad q(0) = q_i$$

* $b = 0$ → exponential decline
* $b = 1$ → harmonic decline
* $0 < b < 1$ → hyperbolic decline (the usual case for unconventional wells)
"""

# ╔═╡ 626c6fdd-f377-420f-9619-25833cfc3e29
using OrdinaryDiffEq, Optimization, OptimizationOptimJL, ForwardDiff, Plots, Random, PlutoUI

# ╔═╡ d827ab6c-b554-4851-bc34-649c187a4dd1
md"""
## 1. The unified Arps ODE

We track two states: the instantaneous rate $q$ and the cumulative production $Q = \int q\,dt$, so a single `solve` call gives us both the forecast and the reserves estimate.
"""

# ╔═╡ ccccf676-fc41-4052-a5fc-896409dd9410
function arps_system!(du, u, p, t)
	q, Q = u
	qi, Di, b = p
	du[1] = -Di * q^(1 + b) / qi^b
	du[2] = q
	return nothing
end

# ╔═╡ 83cea2b1-920b-4481-8f13-d6f663a76f6f
md"""
## 2. Compare the three decline regimes

Drag $b$ to morph continuously from exponential to harmonic decline.
"""

# ╔═╡ 95a4a0be-e902-407b-b172-e7e55aea7eb4
@bind b_demo Slider(0.0:0.1:1.0, default=0.5, show_value=true)

# ╔═╡ bd82d45d-2493-49cf-8ab1-91b8b4a70150
begin
	qi_demo, Di_demo = 1000.0, 0.6   # bbl/d, 1/yr
	demo_prob = ODEProblem(arps_system!, [qi_demo, 0.0], (0.0, 5.0), (qi_demo, Di_demo, b_demo))
	demo_sol = solve(demo_prob, Tsit5(); saveat=0.02)
	plot(demo_sol; idxs=1, xlabel="time (yr)", ylabel="rate q (bbl/d)", label="b = $(b_demo)",
		title="Arps decline family", lw=2)
end

# ╔═╡ 4534bb74-d9e9-420d-8ac7-805e9bd6b5a5
md"""
## 3. Synthetic well data

In practice we don't know $(q_i, D_i, b)$ — we only have noisy monthly production reports and need to back them out. We simulate a "true" hyperbolic well and add realistic (multiplicative) noise.
"""

# ╔═╡ ae7df120-94bc-4392-abdf-f9dadf02c9d5
begin
	rng_og1 = Random.default_rng()
	Random.seed!(rng_og1, 3)

	true_qi, true_Di, true_b = 1200.0, 0.85, 0.6
	t_hist = 1/12:1/12:4.0                      # 4 years of monthly reports

	true_prob = ODEProblem(arps_system!, [true_qi, 0.0], (0.0, 4.0), (true_qi, true_Di, true_b))
	true_sol = solve(true_prob, Tsit5(); saveat=t_hist)
	true_rates = [u[1] for u in true_sol.u]

	observed_rates = true_rates .* (1 .+ 0.06 .* randn(rng_og1, length(true_rates)))
end

# ╔═╡ d05f493f-b081-4a9f-bdd6-08a5ab24f305
scatter(t_hist, observed_rates; label="reported monthly rate", xlabel="time (yr)", ylabel="q (bbl/d)",
	title="Observed well production", yscale=:log10)

# ╔═╡ d031b2d8-481e-4458-b5ad-7cb20928ec6b
md"""
## 4. Fit $(q_i, D_i, b)$ to the data

Decline-curve fitting is nonlinear regression: minimize the mismatch between the model and the reported rates, in log-space (rates are best compared multiplicatively, not additively).
"""

# ╔═╡ 461a4186-5fd6-4da7-82cf-e2c1be4e6e40
function decline_loss(logp, _)
	qi = exp(logp[1]); Di = exp(logp[2]); b = exp(logp[3])
	prob = remake(true_prob; p=(qi, Di, b), u0=[qi, 0.0])
	sol = solve(prob, Tsit5(); saveat=t_hist)
	sol.retcode == ReturnCode.Success || return Inf
	model_rates = [u[1] for u in sol.u]
	return sum(abs2, log.(observed_rates) .- log.(model_rates))
end

# ╔═╡ 78272cda-a29e-4e9b-aba6-3c768391435c
begin
	p0_decline = log.([800.0, 0.5, 0.3])   # deliberately rough initial guess, log-parameterized
	optf_decline = OptimizationFunction(decline_loss, Optimization.AutoForwardDiff())
	optprob_decline = OptimizationProblem(optf_decline, p0_decline)
	fit_decline = Optimization.solve(optprob_decline, OptimizationOptimJL.LBFGS())
end

# ╔═╡ 6bcc40c2-aad4-408a-a7a7-b6e29a5478be
begin
	fitted_qi_1, fitted_Di_1, fitted_b_1 = exp.(fit_decline.u)
	(fitted_qi=fitted_qi_1, fitted_Di=fitted_Di_1, fitted_b=fitted_b_1,
		true_qi=true_qi, true_Di=true_Di, true_b=true_b)
end

# ╔═╡ f946e385-edcd-40cb-a0f5-02c60487e12d
begin
	fitted_p_1 = exp.(fit_decline.u)
	fitted_sol_1 = solve(remake(true_prob; p=Tuple(fitted_p_1), u0=[fitted_p_1[1], 0.0]), Tsit5(); saveat=t_hist)
	scatter(t_hist, observed_rates; label="data", xlabel="time (yr)", ylabel="q (bbl/d)", yscale=:log10)
	plot!(t_hist, [u[1] for u in fitted_sol_1.u]; label="fitted Arps model", lw=2)
end

# ╔═╡ bcc401f8-a98c-4d36-8bd7-1607bf735ab8
md"""
## 5. Forecast to the economic limit and estimate EUR

Once fitted, we forecast forward until the rate drops to an economic limit (say 50 bbl/d) using a `ContinuousCallback` to stop the integration exactly at that crossing, then read off the cumulative production — the well's **Estimated Ultimate Recovery (EUR)**.
"""

# ╔═╡ 1c3b2165-9740-48e6-9fba-494655ac7d43
begin
	economic_limit = 50.0
	condition(u, t, integrator) = u[1] - economic_limit
	affect!(integrator) = terminate!(integrator)
	economic_cb = ContinuousCallback(condition, affect!)

	forecast_prob = remake(true_prob; p=Tuple(fitted_p_1), u0=[fitted_p_1[1], 0.0], tspan=(0.0, 100.0))
	forecast_sol = solve(forecast_prob, Tsit5(); callback=economic_cb)

	economic_life_years = forecast_sol.t[end]
	eur_bbl = forecast_sol.u[end][2] * 365.25   # rate was per day, time in years
end

# ╔═╡ 6ff21cf0-03e5-4e86-acd7-b1c3e5f5484a
plot(forecast_sol; idxs=1, xlabel="time (yr)", ylabel="q (bbl/d)",
	title="Forecast to economic limit ($(round(economic_life_years, digits=1)) yr, EUR ≈ $(round(Int, eur_bbl)) bbl)", lw=2)

# ╔═╡ 49b2b5f1-7640-41b4-898e-836d0edd203c
md"""
## Takeaways

* Writing exponential/harmonic/hyperbolic decline as one autonomous ODE turns three textbook formulas into a single `solve` call.
* Decline-curve fitting is just nonlinear least squares on the ODE's parameters — the same recipe as [`04_parameter_estimation.jl`](../04_parameter_estimation.jl), applied to production data instead of an epidemic curve.
* An event (`ContinuousCallback`) cleanly stops the forecast exactly at the economic limit, giving EUR without hand-picking a stopping time.

Next: [`02_neural_decline_forecasting.jl`](./02_neural_decline_forecasting.jl) tackles wells whose production doesn't follow a clean Arps curve at all.
"""

# ╔═╡ Cell order:
# ╟─4575c6e8-8ec1-4dd1-8f13-a208e28b334f
# ╠═626c6fdd-f377-420f-9619-25833cfc3e29
# ╟─d827ab6c-b554-4851-bc34-649c187a4dd1
# ╠═ccccf676-fc41-4052-a5fc-896409dd9410
# ╟─83cea2b1-920b-4481-8f13-d6f663a76f6f
# ╠═95a4a0be-e902-407b-b172-e7e55aea7eb4
# ╠═bd82d45d-2493-49cf-8ab1-91b8b4a70150
# ╟─4534bb74-d9e9-420d-8ac7-805e9bd6b5a5
# ╠═ae7df120-94bc-4392-abdf-f9dadf02c9d5
# ╠═d05f493f-b081-4a9f-bdd6-08a5ab24f305
# ╟─d031b2d8-481e-4458-b5ad-7cb20928ec6b
# ╠═461a4186-5fd6-4da7-82cf-e2c1be4e6e40
# ╠═78272cda-a29e-4e9b-aba6-3c768391435c
# ╠═6bcc40c2-aad4-408a-a7a7-b6e29a5478be
# ╠═f946e385-edcd-40cb-a0f5-02c60487e12d
# ╟─bcc401f8-a98c-4d36-8bd7-1607bf735ab8
# ╠═1c3b2165-9740-48e6-9fba-494655ac7d43
# ╠═6ff21cf0-03e5-4e86-acd7-b1c3e5f5484a
# ╟─49b2b5f1-7640-41b4-898e-836d0edd203c
