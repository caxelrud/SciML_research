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

# ╔═╡ e073ed55-e7a4-4438-9235-f011e26201d3
md"""
# O&G · 4 — History matching a well test: recovering permeability and skin

A pressure drawdown test is one of the oldest inverse problems in petroleum engineering: flow a well at a constant rate, record the bottomhole pressure, and back out the reservoir permeability $k$ and the near-wellbore skin factor $s$.

For a well producing at constant rate in an infinite-acting reservoir, the semi-log ("MDH") approximation to the radial diffusivity equation's line-source solution is, in standard oilfield units ($q$ in STB/d, $B$ in rb/STB, $\mu$ in cp, $k$ in md, $h$ in ft, $t$ in hours, $\phi$ a fraction, $c_t$ in psi$^{-1}$, $r_w$ in ft, $p$ in psi):

$$p_{wf}(t) = p_i - 162.6\,\frac{qB\mu}{kh}\left[\log_{10}(t) + \log_{10}\!\left(\frac{k}{\phi\mu c_t r_w^2}\right) - 3.23 + 0.87\,s\right]$$

Everything except $k$ and $s$ is either measured or known from core/PVT data — this notebook estimates $k$ and $s$ from a synthetic noisy buildup, the same way an engineer reads a type curve or fits a straight line.
"""

# ╔═╡ f92eede4-d57a-4367-84a3-194ea1eabeaf
using Optimization, OptimizationOptimJL, ForwardDiff, Plots, Random, PlutoUI

# ╔═╡ 7edb12bf-5241-4730-b30e-01cf626acf32
md"""
## 1. The forward model
"""

# ╔═╡ 78a201d3-f891-4e72-a85f-1f744b07b82a
function pwf_model(t, k, s; pi_res=4200.0, q=400.0, B=1.2, μ=1.5, h=60.0, φ=0.18, ct=1.2e-5, rw=0.3)
	Δp = 162.6 * (q * B * μ) / (k * h) * (log10(t) + log10(k / (φ * μ * ct * rw^2)) - 3.23 + 0.87 * s)
	return pi_res - Δp
end

# ╔═╡ 15f86ff2-a506-47fa-873f-c8cda11973d6
md"""
## 2. Play with the model

Skin controls the *extra* pressure drop near the wellbore (damage if $s>0$, stimulation if $s<0$); permeability controls how fast the transient pressure wave dissipates. Drag the sliders and watch the drawdown curve respond.
"""

# ╔═╡ 52b7e245-b4eb-44a6-9dbf-61bfc88eda6d
@bind k_demo Slider(1.0:1.0:200.0, default=50.0, show_value=true)

# ╔═╡ 17ed8291-7953-494f-ad68-89744bd0d554
@bind s_demo Slider(-3.0:0.5:15.0, default=2.0, show_value=true)

# ╔═╡ 3d14d0cc-4846-4150-9a08-3d67793a3cd3
begin
	t_demo = 10 .^ range(-1, 1.5; length=100)   # hours, log-spaced like a real gauge
	plot(t_demo, pwf_model.(t_demo, k_demo, s_demo); xscale=:log10, xlabel="t (hr)", ylabel="pwf (psi)",
		label="k=$(k_demo) md, s=$(s_demo)", title="Simulated drawdown", legend=:bottomleft)
end

# ╔═╡ 0b7c6970-42e0-4db2-914e-aeba2db19089
md"""
## 3. Synthetic buildup data

We generate a noisy gauge record from "true" reservoir properties that we then try to recover blind.
"""

# ╔═╡ 937d7cdf-e339-4a6d-b82b-a99f767a94b5
begin
	rng_og4 = Random.default_rng()
	Random.seed!(rng_og4, 17)

	true_k, true_s = 35.0, 4.5
	t_test = 10 .^ range(-1, 1.3; length=25)     # semi-log analysis window: after wellbore-storage, before boundary effects
	pwf_true = pwf_model.(t_test, true_k, true_s)
	pwf_observed = pwf_true .+ 3.0 .* randn(rng_og4, length(t_test))
end

# ╔═╡ 3c0f2b84-7a3b-42f1-ae48-252b879cd0b4
scatter(log10.(t_test), pwf_observed; label="gauge data", xlabel="log₁₀(t / hr)", ylabel="pwf (psi)",
	title="Semi-log plot (the classic MDH straight line)")

# ╔═╡ 6ffe8115-d10b-4050-97e9-01eb56da1cdd
md"""
## 4. Estimate $k$ and $s$ by nonlinear regression

This is exactly the semi-log straight-line method, just fit by least squares instead of a ruler: the model is linear in $\ln t$, so the fit is extremely well-conditioned.
"""

# ╔═╡ 54819a60-e129-4933-847a-043595289df2
function welltest_loss(logp, _)
	k = exp(logp[1])
	s = logp[2]
	pred = pwf_model.(t_test, k, s)
	return sum(abs2, pwf_observed .- pred)
end

# ╔═╡ 78fc7f87-3e4d-4e78-978f-71c90cec6fb9
begin
	p0_welltest = [log(10.0), 0.0]
	optf_welltest = OptimizationFunction(welltest_loss, Optimization.AutoForwardDiff())
	optprob_welltest = OptimizationProblem(optf_welltest, p0_welltest)
	fit_welltest = Optimization.solve(optprob_welltest, OptimizationOptimJL.LBFGS())
end

# ╔═╡ 995e5eb4-cab4-4d1a-9f9b-806bb96b7961
(fitted_k=exp(fit_welltest.u[1]), fitted_s=fit_welltest.u[2], true_k=true_k, true_s=true_s)

# ╔═╡ 47dd8f59-d055-4d7e-81f1-a6310af8058f
begin
	scatter(log10.(t_test), pwf_observed; label="data", xlabel="log₁₀(t / hr)", ylabel="pwf (psi)")
	fitted_k_1 = exp(fit_welltest.u[1])
	plot!(log10.(t_test), pwf_model.(t_test, fitted_k_1, fit_welltest.u[2]); label="fitted model", lw=2,
		title="Recovered k = $(round(fitted_k_1, digits=1)) md, s = $(round(fit_welltest.u[2], digits=2))")
end

# ╔═╡ 1328920d-e995-436b-a662-6a2bbbbec6c6
md"""
## Takeaways

* The semi-log straight-line method used throughout well-test analysis is nothing more than fitting a known closed-form model to data — `Optimization.jl` does it without needing to identify the straight-line region by eye.
* Because the model here is an explicit formula rather than an ODE solve, no sensitivity-analysis machinery is needed at all — plain `ForwardDiff` through the formula is enough.
* The same nonlinear-regression pattern extends directly to more complex well-test models (dual-porosity, finite-conductivity fractures, bounded reservoirs) — only `pwf_model` needs to change; the fitting code stays identical.
"""

# ╔═╡ Cell order:
# ╟─e073ed55-e7a4-4438-9235-f011e26201d3
# ╠═f92eede4-d57a-4367-84a3-194ea1eabeaf
# ╟─7edb12bf-5241-4730-b30e-01cf626acf32
# ╠═78a201d3-f891-4e72-a85f-1f744b07b82a
# ╟─15f86ff2-a506-47fa-873f-c8cda11973d6
# ╠═52b7e245-b4eb-44a6-9dbf-61bfc88eda6d
# ╠═17ed8291-7953-494f-ad68-89744bd0d554
# ╠═3d14d0cc-4846-4150-9a08-3d67793a3cd3
# ╟─0b7c6970-42e0-4db2-914e-aeba2db19089
# ╠═937d7cdf-e339-4a6d-b82b-a99f767a94b5
# ╠═3c0f2b84-7a3b-42f1-ae48-252b879cd0b4
# ╟─6ffe8115-d10b-4050-97e9-01eb56da1cdd
# ╠═54819a60-e129-4933-847a-043595289df2
# ╠═78fc7f87-3e4d-4e78-978f-71c90cec6fb9
# ╠═995e5eb4-cab4-4d1a-9f9b-806bb96b7961
# ╠═47dd8f59-d055-4d7e-81f1-a6310af8058f
# ╟─1328920d-e995-436b-a662-6a2bbbbec6c6
