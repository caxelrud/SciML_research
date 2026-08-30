### A Pluto.jl notebook ###
# 0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ b5baf838-1d22-4fb0-a8ec-97fe8cb13abd
md"""
# 7 · Bayesian parameter estimation with Turing.jl

[`04_parameter_estimation.jl`](./04_parameter_estimation.jl) found a single *best-fit* point for the SIR model's parameters. That answers "what values fit best?" but not "how sure are we?" — with noisy, limited data, many nearby parameter values could explain the data almost as well.

**Bayesian inference** answers the second question directly: instead of a point estimate, it returns a full *posterior distribution* over the parameters, obtained by combining a prior belief with the likelihood of the observed data under the ODE model. We sample that posterior with `Turing.jl`'s No-U-Turn Sampler (NUTS), reusing exactly the same `OrdinaryDiffEq.jl` forward model as the point-estimate notebook — only the fitting method changes.
"""

# ╔═╡ bbd8cb57-feb2-4ea0-b715-f2acf1ee205d
using Turing, OrdinaryDiffEq, Random, Statistics, Plots

# ╔═╡ 25676c29-62a1-4879-9d94-7a3bdfc73a73
md"""
## 1. The model and synthetic data

Same SIR epidemic model and the same synthetic noisy observations of $I(t)$ as in [`04_parameter_estimation.jl`](./04_parameter_estimation.jl).
"""

# ╔═╡ be445daf-c756-456d-bcf6-81225d585106
function sir!(du, u, p, t)
	S, I, R = u
	β, γ = p
	du[1] = -β * S * I
	du[2] = β * S * I - γ * I
	du[3] = γ * I
	return nothing
end

# ╔═╡ beeaad8d-a175-4ee4-87ce-b8216a74c8ef
begin
	rng7 = Random.default_rng()
	Random.seed!(rng7, 21)

	true_β7, true_γ7 = 0.4, 0.1
	u0_sir7 = [0.99, 0.01, 0.0]
	tspan_sir7 = (0.0, 60.0)
	tsteps_sir7 = range(tspan_sir7[1], tspan_sir7[2]; length=16)

	sir_prob7 = ODEProblem(sir!, u0_sir7, tspan_sir7, [true_β7, true_γ7])
	sir_sol7 = solve(sir_prob7, Tsit5(); saveat=tsteps_sir7)

	noise_level7 = 0.02
	observed_I = Array(sir_sol7)[2, :] .+ noise_level7 .* randn(rng7, length(tsteps_sir7))
end

# ╔═╡ 3adc1e10-efc0-40a6-b8f5-5391c5f494ff
scatter(tsteps_sir7, observed_I; label="observed I(t)", xlabel="t", ylabel="I(t)", title="Data given to the Bayesian model")

# ╔═╡ 1c6b4bf0-4a9a-4986-b4a3-1a99b3aae3ab
md"""
## 2. The Bayesian model

`@model` declares priors on $\beta$, $\gamma$, and the observation noise $\sigma$, solves the ODE for a candidate parameter draw, and scores how likely the observed $I(t)$ is under a normal observation model centered on the simulated trajectory. `Turing.jl` differentiates straight through the `OrdinaryDiffEq.jl` solve (forward-mode, the same mechanism used for the fits in the O&G notebooks) to drive NUTS.
"""

# ╔═╡ 9fd9d6bd-e1a9-40b1-a6b8-58de522dda6d
Turing.@model function sir_bayesian(data, prob, tsteps)
	β ~ truncated(Normal(0.5, 0.5); lower=0.0, upper=3.0)
	γ ~ truncated(Normal(0.2, 0.2); lower=0.0, upper=3.0)
	σ ~ truncated(Normal(0.05, 0.05); lower=0.001, upper=1.0)

	sol = solve(prob, Tsit5(); p=[β, γ], saveat=tsteps)
	if sol.retcode != ReturnCode.Success
		Turing.@addlogprob! -Inf
		return nothing
	end

	predicted_I = Array(sol)[2, :]
	for i in eachindex(tsteps)
		data[i] ~ Normal(predicted_I[i], σ)
	end
	return nothing
end

# ╔═╡ b7d5eb9e-1af2-481b-992c-b775f67f504f
md"""
## 3. Sample the posterior
"""

# ╔═╡ 4d1d7167-3ac8-40b4-be1a-53dd7ff35002
begin
	bayes_model = sir_bayesian(observed_I, sir_prob7, tsteps_sir7)
	chain = sample(bayes_model, NUTS(0.65), 1000)
end

# ╔═╡ 2c59adca-c2c7-406e-b615-3cb82652e977
md"""
## 4. Posterior distributions vs. the true values

Unlike the single point estimate from `Optimization.jl`, NUTS returns a full sample from the posterior — its spread *is* the uncertainty.
"""

# ╔═╡ 96daa50d-a5c9-49d7-8907-773f42358943
begin
	β_samples = vec(Array(chain[:β]))
	γ_samples = vec(Array(chain[:γ]))

	p1 = histogram(β_samples; label="posterior", xlabel="β", title="β  (true = $(true_β7))", normalize=true)
	vline!(p1, [true_β7]; label="true β", lw=3)

	p2 = histogram(γ_samples; label="posterior", xlabel="γ", title="γ  (true = $(true_γ7))", normalize=true)
	vline!(p2, [true_γ7]; label="true γ", lw=3)

	plot(p1, p2; layout=(1, 2), size=(700, 300))
end

# ╔═╡ a87fc3c4-c456-431a-bf62-69e798ce299c
(β_mean=mean(β_samples), β_std=std(β_samples), γ_mean=mean(γ_samples), γ_std=std(γ_samples))

# ╔═╡ 4df6b695-af03-47c8-9b41-1cdc89aa26c0
md"""
## 5. Posterior predictive check

Solving the ODE forward for many posterior draws (instead of just the best fit) gives a *band* of plausible trajectories — a direct visualization of forecast uncertainty.
"""

# ╔═╡ fab14908-6c89-4a5e-aaf3-08847eec1277
begin
	n_draws = 60
	draw_idxs = rand(rng7, 1:length(β_samples), n_draws)
	t_fine = range(tspan_sir7[1], tspan_sir7[2]; length=200)

	plt = plot(; xlabel="t", ylabel="I(t)", title="Posterior predictive: $(n_draws) draws", legend=:topright)
	for i in draw_idxs
		draw_sol = solve(remake(sir_prob7; p=[β_samples[i], γ_samples[i]]), Tsit5(); saveat=t_fine)
		plot!(plt, t_fine, Array(draw_sol)[2, :]; color=:steelblue, alpha=0.08, label=false)
	end
	scatter!(plt, tsteps_sir7, observed_I; color=:black, label="data")
	plt
end

# ╔═╡ 20436856-32e2-41d5-b60c-7c6b364a5030
md"""
## Takeaways

* A Bayesian model is the same forward simulation as before, wrapped in priors and a likelihood — `Turing.jl` handles the sampling machinery, not the ODE.
* The posterior's spread is honest uncertainty: it widens automatically with noisier or sparser data, something a single point estimate from [`04_parameter_estimation.jl`](./04_parameter_estimation.jl) cannot express on its own.
* The posterior *predictive* band (many forward solves, one per posterior draw) is the natural way to report a forecast with uncertainty, rather than a single curve.
"""

# ╔═╡ Cell order:
# ╟─b5baf838-1d22-4fb0-a8ec-97fe8cb13abd
# ╠═bbd8cb57-feb2-4ea0-b715-f2acf1ee205d
# ╟─25676c29-62a1-4879-9d94-7a3bdfc73a73
# ╠═be445daf-c756-456d-bcf6-81225d585106
# ╠═beeaad8d-a175-4ee4-87ce-b8216a74c8ef
# ╠═3adc1e10-efc0-40a6-b8f5-5391c5f494ff
# ╟─1c6b4bf0-4a9a-4986-b4a3-1a99b3aae3ab
# ╠═9fd9d6bd-e1a9-40b1-a6b8-58de522dda6d
# ╟─b7d5eb9e-1af2-481b-992c-b775f67f504f
# ╠═4d1d7167-3ac8-40b4-be1a-53dd7ff35002
# ╟─2c59adca-c2c7-406e-b615-3cb82652e977
# ╠═96daa50d-a5c9-49d7-8907-773f42358943
# ╠═a87fc3c4-c456-431a-bf62-69e798ce299c
# ╟─4df6b695-af03-47c8-9b41-1cdc89aa26c0
# ╠═fab14908-6c89-4a5e-aaf3-08847eec1277
# ╟─20436856-32e2-41d5-b60c-7c6b364a5030
