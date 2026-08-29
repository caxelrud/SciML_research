### A Pluto.jl notebook ###
# 0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ adc9f0e4-240d-48be-a8bc-094af42a62f9
md"""
# 3 · Universal Differential Equations (UDE)

A **Universal Differential Equation** mixes *known* mechanistic terms with an *unknown* term represented by a neural network, and learns the unknown part from data. This is the key idea behind Rackauckas et al. (2020), *Universal Differential Equations for Scientific Machine Learning*.

We use the Lotka–Volterra predator–prey model:

$$\dot x = \alpha x - \beta x y, \qquad \dot y = -\gamma y + \delta x y$$

We pretend we only know the growth term $\alpha x$ and decay term $-\gamma y$, and let a neural network $U_\theta(x, y)$ learn the *interaction* terms $-\beta xy$ and $\delta xy$ directly from trajectory data.
"""

# ╔═╡ b8b04249-2ae6-420d-80b2-bbb92437fcd7
using Lux, OrdinaryDiffEq, SciMLSensitivity, Optimization, OptimizationOptimisers, ComponentArrays, Random, Plots

# ╔═╡ eb6b4429-ce32-4332-898a-51c93516a608
md"""
## 1. Ground-truth data from the full Lotka–Volterra model
"""

# ╔═╡ c68589cf-17a7-4885-94d3-709886fce2ec
function lotka_volterra!(du, u, p, t)
	x, y = u
	α, β, γ, δ = p
	du[1] = α * x - β * x * y
	du[2] = -γ * y + δ * x * y
	return nothing
end

# ╔═╡ 9058c6cf-7879-492e-a455-ef1400379a8c
begin
	rng3 = Random.default_rng()
	Random.seed!(rng3, 42)

	true_p = (1.3, 0.9, 0.8, 1.8)   # α, β, γ, δ
	u0_lv = [0.44, 4.6]
	tspan_lv = (0.0, 8.0)
	tsteps_lv = range(tspan_lv[1], tspan_lv[2]; length=60)

	lv_prob = ODEProblem(lotka_volterra!, u0_lv, tspan_lv, true_p)
	lv_sol = solve(lv_prob, Tsit5(); saveat=tsteps_lv)
	lv_data = Array(lv_sol)
end

# ╔═╡ c2737565-0bb9-4ae1-817e-915732ab29d0
plot(tsteps_lv, lv_data'; label=["prey x" "predator y"], xlabel="t", title="Lotka–Volterra ground truth")

# ╔═╡ e5a4f63d-c819-45d3-8e88-11d71467105b
md"""
## 2. The hybrid (universal) model

We keep the known linear growth/decay terms $\alpha x$ and $-\gamma y$ exactly, and replace the *unknown* interaction terms with a small neural network $U_\theta(x, y) \in \mathbb{R}^2$:

$$\dot x = \alpha x + U_\theta(x, y)_1, \qquad \dot y = -\gamma y + U_\theta(x, y)_2$$
"""

# ╔═╡ 615615da-9347-4fd9-830b-df8ff25a7a66
begin
	interaction_nn = Chain(Dense(2, 8, tanh), Dense(8, 8, tanh), Dense(8, 2))
	nn_ps_init, nn_st = Lux.setup(rng3, interaction_nn)
	nn_ps_init = ComponentArray(nn_ps_init)
end

# ╔═╡ 502a47de-456b-4b2b-9c31-514e595ca7c7
begin
	# only the growth/decay rates are assumed known
	α_known = 1.3
	γ_known = 0.8
end

# ╔═╡ a0015846-d10d-41a1-bcb8-376cf56b4557
function ude_dynamics!(du, u, p, t)
	x, y = u
	û, _ = interaction_nn(u, p, nn_st)
	du[1] = α_known * x + û[1]
	du[2] = -γ_known * y + û[2]
	return nothing
end

# ╔═╡ cb20d13d-4b4c-4c42-8dfd-c59aa7cb72d9
ude_prob = ODEProblem(ude_dynamics!, u0_lv, tspan_lv)

# ╔═╡ e028b9f9-8d07-4cb8-b185-2a91548eff8e
md"""
## 3. Train the neural interaction term against the data
"""

# ╔═╡ 0b9ace54-41d2-4496-ad89-c784595a22ea
function predict_ude(p)
	solve(ude_prob, Tsit5(); p=p, saveat=tsteps_lv,
		sensealg=InterpolatingAdjoint(; autojacvec=ZygoteVJP())) |> Array
end

# ╔═╡ 00c66f29-3723-47ee-aaf2-a21892ba9101
function loss_ude(p, _)
	pred = predict_ude(p)
	size(pred) == size(lv_data) || return Inf
	return sum(abs2, lv_data .- pred)
end

# ╔═╡ 61d1bfdd-7187-464d-a900-5ca8b5b6acd5
begin
	adtype3 = Optimization.AutoZygote()
	optf3 = OptimizationFunction(loss_ude, adtype3)
	optprob3 = OptimizationProblem(optf3, nn_ps_init)

	ude_losses = Float64[]
	ude_callback = function (state, loss)
		push!(ude_losses, loss)
		return false
	end

	res_ude = Optimization.solve(optprob3, OptimizationOptimisers.Adam(0.02); callback=ude_callback, maxiters=500)
end

# ╔═╡ e7558cdd-6545-4909-9e9a-27763622d5fc
plot(ude_losses; xlabel="iteration", ylabel="loss", yscale=:log10, label="UDE training loss")

# ╔═╡ 3bf34956-c252-46de-9390-111255f07f8d
md"""
## 4. Does the learned model reproduce the trajectories?
"""

# ╔═╡ 8068f2d4-c9e9-4750-8bab-5169e5e7146c
begin
	ude_fit = predict_ude(res_ude.u)
	plot(tsteps_lv, lv_data'; label=["data x" "data y"], lw=2)
	plot!(tsteps_lv, ude_fit'; label=["UDE x" "UDE y"], lw=2, ls=:dash)
end

# ╔═╡ d189cc54-1744-4d9a-a7cc-55dcce8926fc
md"""
## 5. Inspect the learned interaction term

The true missing interaction terms are $-\beta xy$ and $\delta xy$. We can compare them against what the trained network $U_\theta$ actually learned, on a grid of $(x, y)$ values.
"""

# ╔═╡ 1acd09ba-f73c-45a1-b7fb-87e25cad6e93
begin
	test_x, test_y = 1.5, 1.0
	learned_interaction, _ = interaction_nn([test_x, test_y], res_ude.u, nn_st)
	true_interaction = [-true_p[2] * test_x * test_y, true_p[4] * test_x * test_y]
	(learned=learned_interaction, true_term=true_interaction)
end

# ╔═╡ 58510873-2b91-4df8-a46a-b21f1cb2180c
md"""
## Takeaways

* Only the physics you *don't* trust needs to be replaced by a neural network — everything you already know (conservation laws, known rates) stays exactly as an equation.
* Because the unknown term is low-dimensional and structured, the UDE typically needs far less data than a fully black-box neural ODE.
* Once $U_\theta$ is trained, its input/output pairs can be fed to a **sparse regression** method (SINDy, see [`06_sindy_model_discovery.jl`](./06_sindy_model_discovery.jl)) to recover a symbolic formula such as $-\beta xy$ — turning the learned network back into an interpretable equation.
"""

# ╔═╡ Cell order:
# ╟─adc9f0e4-240d-48be-a8bc-094af42a62f9
# ╠═b8b04249-2ae6-420d-80b2-bbb92437fcd7
# ╟─eb6b4429-ce32-4332-898a-51c93516a608
# ╠═c68589cf-17a7-4885-94d3-709886fce2ec
# ╠═9058c6cf-7879-492e-a455-ef1400379a8c
# ╠═c2737565-0bb9-4ae1-817e-915732ab29d0
# ╟─e5a4f63d-c819-45d3-8e88-11d71467105b
# ╠═615615da-9347-4fd9-830b-df8ff25a7a66
# ╠═502a47de-456b-4b2b-9c31-514e595ca7c7
# ╠═a0015846-d10d-41a1-bcb8-376cf56b4557
# ╠═cb20d13d-4b4c-4c42-8dfd-c59aa7cb72d9
# ╟─e028b9f9-8d07-4cb8-b185-2a91548eff8e
# ╠═0b9ace54-41d2-4496-ad89-c784595a22ea
# ╠═00c66f29-3723-47ee-aaf2-a21892ba9101
# ╠═61d1bfdd-7187-464d-a900-5ca8b5b6acd5
# ╠═e7558cdd-6545-4909-9e9a-27763622d5fc
# ╟─3bf34956-c252-46de-9390-111255f07f8d
# ╠═8068f2d4-c9e9-4750-8bab-5169e5e7146c
# ╟─d189cc54-1744-4d9a-a7cc-55dcce8926fc
# ╠═1acd09ba-f73c-45a1-b7fb-87e25cad6e93
# ╟─58510873-2b91-4df8-a46a-b21f1cb2180c
