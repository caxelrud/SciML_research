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

# ╔═╡ 5330334e-2bdf-4955-94f7-b46befdc6ff3
md"""
# 4 · Parameter estimation & inverse problems

A very common SciML task: given noisy experimental observations and a mechanistic ODE model with *unknown* parameters, find the parameter values that best explain the data. This is the classic **inverse problem**, solved here by turning it into an optimization problem: minimize the mismatch between simulated and observed trajectories over the ODE parameters.
"""

# ╔═╡ 2827adcf-4842-4240-b83f-4ff96171854c
using OrdinaryDiffEq, SciMLSensitivity, Optimization, OptimizationOptimJL, Plots, Random, PlutoUI

# ╔═╡ c560a03b-eef1-4e18-81af-a9635bb7bf0d
md"""
## 1. The model: a SIR epidemic

$$\dot S = -\beta S I, \qquad \dot I = \beta S I - \gamma I, \qquad \dot R = \gamma I$$

$\beta$ is the infection rate and $\gamma$ the recovery rate. In a real setting these come from data fitting rather than being known constants.
"""

# ╔═╡ 945519d8-bb04-4394-ad2e-6ec139a59ff3
function sir!(du, u, p, t)
	S, I, R = u
	β, γ = p
	du[1] = -β * S * I
	du[2] = β * S * I - γ * I
	du[3] = γ * I
	return nothing
end

# ╔═╡ 8fa7d8d2-ae85-4445-baa7-3424a96fde93
md"""
## 2. Generate noisy "observed" data

We simulate with known true parameters, then add Gaussian observation noise — mimicking real measurement error.
"""

# ╔═╡ ce790b23-ccbc-45b4-8c31-b488bfcd21b8
begin
	rng4 = Random.default_rng()
	Random.seed!(rng4, 7)

	true_β, true_γ = 0.4, 0.1
	u0_sir = [0.99, 0.01, 0.0]
	tspan_sir = (0.0, 60.0)
	tsteps_sir = range(tspan_sir[1], tspan_sir[2]; length=30)

	sir_prob = ODEProblem(sir!, u0_sir, tspan_sir, [true_β, true_γ])
	sir_sol = solve(sir_prob, Tsit5(); saveat=tsteps_sir)

	noise_level = 0.02
	sir_data = Array(sir_sol) .+ noise_level .* randn(rng4, size(Array(sir_sol)))
end

# ╔═╡ 596dd851-a351-4af3-b310-772ac2079036
begin
	scatter(tsteps_sir, sir_data[2, :]; label="observed I(t)", xlabel="t", ylabel="fraction of population")
	plot!(tsteps_sir, Array(sir_sol)[2, :]; label="true I(t)", lw=2)
end

# ╔═╡ 164aa1f2-3d3b-4208-9f14-c5622281591e
md"""
## 3. Loss landscape

Before fitting, it helps to see how sensitive the mismatch is to each parameter. Drag the sliders below and watch the simulated curve chase the noisy data.
"""

# ╔═╡ 5aeff909-49b5-40f0-ba38-1a26565c9cf6
@bind β_guess Slider(0.1:0.02:0.8, default=0.3, show_value=true)

# ╔═╡ a86e4cf4-e82e-4385-aa76-96d39bec0f01
@bind γ_guess Slider(0.02:0.02:0.5, default=0.2, show_value=true)

# ╔═╡ 7a4481d4-db98-4a5e-8238-5a33018e3540
begin
	guess_prob = remake(sir_prob; p=[β_guess, γ_guess])
	guess_sol = solve(guess_prob, Tsit5(); saveat=tsteps_sir)
	guess_loss = sum(abs2, sir_data .- Array(guess_sol))

	scatter(tsteps_sir, sir_data[2, :]; label="data", xlabel="t", ylabel="I(t)")
	plot!(tsteps_sir, Array(guess_sol)[2, :]; label="guess (β=$(β_guess), γ=$(γ_guess))", lw=2,
		title="sum of squared error = $(round(guess_loss, digits=4))")
end

# ╔═╡ f1697c44-de81-4ef7-8d39-94cc803ca5e6
md"""
## 4. Fit the parameters automatically

Rather than tuning sliders by hand, we let `Optimization.jl` minimize the sum-of-squares loss directly. Because the loss only needs a forward `solve`, we can use a derivative-free or a gradient-based local optimizer — here we use `Optim.jl`'s Nelder–Mead via `OptimizationOptimJL`.
"""

# ╔═╡ 3b1e9915-c6a1-439c-96d6-5a30f07e2cd2
function sir_loss(p, _)
	prob = remake(sir_prob; p=p)
	sol = solve(prob, Tsit5(); saveat=tsteps_sir)
	sol.retcode == ReturnCode.Success || return Inf
	return sum(abs2, sir_data .- Array(sol))
end

# ╔═╡ 64a83a78-9cd8-44c6-ab36-936de33bb71f
begin
	p0 = [0.2, 0.3]   # initial guess, deliberately off from the truth
	optf4 = OptimizationFunction(sir_loss)
	optprob4 = OptimizationProblem(optf4, p0)
	fit_res = Optimization.solve(optprob4, OptimizationOptimJL.NelderMead())
end

# ╔═╡ b7386ffe-11c5-4e88-92a3-094aeeffe91e
(fitted_β=fit_res.u[1], fitted_γ=fit_res.u[2], true_β=true_β, true_γ=true_γ)

# ╔═╡ 089b8eb8-e211-4ad5-919c-33b6429dedc5
begin
	fitted_sol = solve(remake(sir_prob; p=fit_res.u), Tsit5(); saveat=tsteps_sir)
	scatter(tsteps_sir, sir_data[2, :]; label="data", xlabel="t", ylabel="I(t)")
	plot!(tsteps_sir, Array(fitted_sol)[2, :]; label="fitted model", lw=2, title="Recovered SIR fit")
end

# ╔═╡ 3eb00ec2-1b5e-462a-b159-bad056bd931d
md"""
## Takeaways

* Parameter estimation is just optimization: wrap `solve` in a loss function and hand it to `Optimization.jl`.
* `remake(prob; p=...)` avoids rebuilding the `ODEProblem` from scratch on every loss evaluation — a small but important performance detail.
* For harder, higher-dimensional or multimodal problems, swap in a global optimizer (`OptimizationBBO`, `OptimizationCMAEvolutionStrategy`) or a Bayesian approach (`Turing.jl` on top of the same forward model) with no change to the ODE code itself.
"""

# ╔═╡ Cell order:
# ╟─5330334e-2bdf-4955-94f7-b46befdc6ff3
# ╠═2827adcf-4842-4240-b83f-4ff96171854c
# ╟─c560a03b-eef1-4e18-81af-a9635bb7bf0d
# ╠═945519d8-bb04-4394-ad2e-6ec139a59ff3
# ╟─8fa7d8d2-ae85-4445-baa7-3424a96fde93
# ╠═ce790b23-ccbc-45b4-8c31-b488bfcd21b8
# ╠═596dd851-a351-4af3-b310-772ac2079036
# ╟─164aa1f2-3d3b-4208-9f14-c5622281591e
# ╠═5aeff909-49b5-40f0-ba38-1a26565c9cf6
# ╠═a86e4cf4-e82e-4385-aa76-96d39bec0f01
# ╠═7a4481d4-db98-4a5e-8238-5a33018e3540
# ╟─f1697c44-de81-4ef7-8d39-94cc803ca5e6
# ╠═3b1e9915-c6a1-439c-96d6-5a30f07e2cd2
# ╠═64a83a78-9cd8-44c6-ab36-936de33bb71f
# ╠═b7386ffe-11c5-4e88-92a3-094aeeffe91e
# ╠═089b8eb8-e211-4ad5-919c-33b6429dedc5
# ╟─3eb00ec2-1b5e-462a-b159-bad056bd931d
