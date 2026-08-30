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

# ╔═╡ 74002011-f764-4fee-9971-801a3d343209
md"""
# 4 · Parameter estimation & inverse problems

A very common SciML task: given noisy experimental observations and a mechanistic ODE model with *unknown* parameters, find the parameter values that best explain the data. This is the classic **inverse problem**, solved here by turning it into an optimization problem: minimize the mismatch between simulated and observed trajectories over the ODE parameters.
"""

# ╔═╡ b2a5e353-67f4-4769-8e49-3c9197398592
using OrdinaryDiffEq, Optimization, OptimizationOptimJL, Plots, Random, PlutoUI

# ╔═╡ 1761c94b-11c7-4729-93cb-b485f326fbcc
md"""
## 1. The model: a SIR epidemic

$$\dot S = -\beta S I, \qquad \dot I = \beta S I - \gamma I, \qquad \dot R = \gamma I$$

$\beta$ is the infection rate and $\gamma$ the recovery rate. In a real setting these come from data fitting rather than being known constants.
"""

# ╔═╡ 51d548e3-e962-4449-a512-c69600f8b3f5
function sir!(du, u, p, t)
	S, I, R = u
	β, γ = p
	du[1] = -β * S * I
	du[2] = β * S * I - γ * I
	du[3] = γ * I
	return nothing
end

# ╔═╡ 71c19bdb-fe3a-461a-893e-192227685dc6
md"""
## 2. Generate noisy "observed" data

We simulate with known true parameters, then add Gaussian observation noise — mimicking real measurement error.
"""

# ╔═╡ 940aea8b-ffbd-466a-acfb-39af74bc7555
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

# ╔═╡ 37ba20ac-ce83-4bc5-9719-4670881398d4
begin
	scatter(tsteps_sir, sir_data[2, :]; label="observed I(t)", xlabel="t", ylabel="fraction of population")
	plot!(tsteps_sir, Array(sir_sol)[2, :]; label="true I(t)", lw=2)
end

# ╔═╡ 6bba8741-a2ad-4ce4-a67b-5d6d845add24
md"""
## 3. Loss landscape

Before fitting, it helps to see how sensitive the mismatch is to each parameter. Drag the sliders below and watch the simulated curve chase the noisy data.
"""

# ╔═╡ bbb3e22e-ed62-45d8-b8ed-75579d876fac
@bind β_guess Slider(0.1:0.02:0.8, default=0.3, show_value=true)

# ╔═╡ 071353b0-5e87-4521-a35b-eae8e57c1eaf
@bind γ_guess Slider(0.02:0.02:0.5, default=0.2, show_value=true)

# ╔═╡ f0dec18f-2875-4ee5-8123-63f69836a66c
begin
	guess_prob = remake(sir_prob; p=[β_guess, γ_guess])
	guess_sol = solve(guess_prob, Tsit5(); saveat=tsteps_sir)
	guess_loss = sum(abs2, sir_data .- Array(guess_sol))

	scatter(tsteps_sir, sir_data[2, :]; label="data", xlabel="t", ylabel="I(t)")
	plot!(tsteps_sir, Array(guess_sol)[2, :]; label="guess (β=$(β_guess), γ=$(γ_guess))", lw=2,
		title="sum of squared error = $(round(guess_loss, digits=4))")
end

# ╔═╡ 1a0f266f-2ce0-47b1-88c5-e21dcd3f0e95
md"""
## 4. Fit the parameters automatically

Rather than tuning sliders by hand, we let `Optimization.jl` minimize the sum-of-squares loss directly. Because the loss only needs a forward `solve`, we can use a derivative-free or a gradient-based local optimizer — here we use `Optim.jl`'s Nelder–Mead via `OptimizationOptimJL`.
"""

# ╔═╡ cc921c14-4b1c-4bf7-ade1-195218b57018
function sir_loss(p, _)
	prob = remake(sir_prob; p=p)
	sol = solve(prob, Tsit5(); saveat=tsteps_sir)
	sol.retcode == ReturnCode.Success || return Inf
	return sum(abs2, sir_data .- Array(sol))
end

# ╔═╡ 74955415-3dbf-4f2a-8a8d-f0498f49ae37
begin
	p0 = [0.2, 0.3]   # initial guess, deliberately off from the truth
	optf4 = OptimizationFunction(sir_loss)
	optprob4 = OptimizationProblem(optf4, p0)
	fit_res = Optimization.solve(optprob4, OptimizationOptimJL.NelderMead())
end

# ╔═╡ 09169668-bc8c-49f1-a29b-80fa96b1b3d0
(fitted_β=fit_res.u[1], fitted_γ=fit_res.u[2], true_β=true_β, true_γ=true_γ)

# ╔═╡ 04e6efb9-35bc-42b6-8a96-d5cd4fd92100
begin
	fitted_sol = solve(remake(sir_prob; p=fit_res.u), Tsit5(); saveat=tsteps_sir)
	scatter(tsteps_sir, sir_data[2, :]; label="data", xlabel="t", ylabel="I(t)")
	plot!(tsteps_sir, Array(fitted_sol)[2, :]; label="fitted model", lw=2, title="Recovered SIR fit")
end

# ╔═╡ 018add5c-f619-4e6d-ad91-cacfb5d1d2e9
md"""
## Takeaways

* Parameter estimation is just optimization: wrap `solve` in a loss function and hand it to `Optimization.jl`.
* `remake(prob; p=...)` avoids rebuilding the `ODEProblem` from scratch on every loss evaluation — a small but important performance detail.
* For harder, higher-dimensional or multimodal problems, swap in a global optimizer (`OptimizationBBO`, `OptimizationCMAEvolutionStrategy`) or a Bayesian approach (`Turing.jl` on top of the same forward model) with no change to the ODE code itself.
"""

# ╔═╡ Cell order:
# ╟─74002011-f764-4fee-9971-801a3d343209
# ╠═b2a5e353-67f4-4769-8e49-3c9197398592
# ╟─1761c94b-11c7-4729-93cb-b485f326fbcc
# ╠═51d548e3-e962-4449-a512-c69600f8b3f5
# ╟─71c19bdb-fe3a-461a-893e-192227685dc6
# ╠═940aea8b-ffbd-466a-acfb-39af74bc7555
# ╠═37ba20ac-ce83-4bc5-9719-4670881398d4
# ╟─6bba8741-a2ad-4ce4-a67b-5d6d845add24
# ╠═bbb3e22e-ed62-45d8-b8ed-75579d876fac
# ╠═071353b0-5e87-4521-a35b-eae8e57c1eaf
# ╠═f0dec18f-2875-4ee5-8123-63f69836a66c
# ╟─1a0f266f-2ce0-47b1-88c5-e21dcd3f0e95
# ╠═cc921c14-4b1c-4bf7-ade1-195218b57018
# ╠═74955415-3dbf-4f2a-8a8d-f0498f49ae37
# ╠═09169668-bc8c-49f1-a29b-80fa96b1b3d0
# ╠═04e6efb9-35bc-42b6-8a96-d5cd4fd92100
# ╟─018add5c-f619-4e6d-ad91-cacfb5d1d2e9
