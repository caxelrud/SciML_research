### A Pluto.jl notebook ###
# 0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ 15b5f862-02fe-42eb-8c6f-edf193d60a79
md"""
# 2 · Neural ODEs

A **Neural ODE** replaces the right-hand side of a differential equation with a neural network:

$$\dot u = f_\theta(u, t)$$

where $f_\theta$ is a small neural network with trainable parameters $\theta$. Training means: simulate the ODE forward, compare to data, and backpropagate *through the ODE solver* to update $\theta$. This notebook trains a neural ODE to reproduce a 2D spiral trajectory, following the classic example from Chen et al. (2018), *Neural Ordinary Differential Equations*.
"""

# ╔═╡ cd436a19-abad-49c8-99ea-9bea52c61ce4
using Lux, OrdinaryDiffEq, SciMLSensitivity, Optimization, OptimizationOptimisers, ComponentArrays, Random, Plots

# ╔═╡ b217208c-c9b2-431b-942b-3f9f4494c302
md"""
## 1. Generate synthetic "ground truth" data

We pretend we don't know the true dynamics and only observe noiseless samples of a 2D spiral produced by a simple linear ODE $\dot u = Au$.
"""

# ╔═╡ da941467-3549-4cfd-9a88-1590735978fe
begin
	rng = Random.default_rng()
	Random.seed!(rng, 1)

	true_A = [-0.1 2.0; -2.0 -0.1]
	true_odefunc(u, p, t) = true_A * (u .^ 3)   # nonlinear spiral (cubic term)

	u0_true = Float32[2.0, 0.0]
	tspan = (0.0f0, 1.5f0)
	tsteps = range(tspan[1], tspan[2]; length=40)

	true_prob = ODEProblem(true_odefunc, u0_true, tspan)
	true_sol = solve(true_prob, Tsit5(); saveat=tsteps)
	ode_data = Array(true_sol)
end

# ╔═╡ 2795b34b-5dfd-4037-b1cf-c008739f7540
scatter(tsteps, ode_data[1, :]; label="u₁ (data)", xlabel="t", title="Observed spiral trajectory (state 1)")

# ╔═╡ b85ddfb2-4919-4982-8b02-9af675ebeed7
md"""
## 2. Define the neural ODE

Instead of the (unknown, in a real problem) true dynamics, we parameterize $f_\theta$ with a small `Lux` multilayer perceptron and wrap it in an `ODEProblem`.
"""

# ╔═╡ f7eeda23-9a2a-417f-bcc1-5751c0e27cbb
begin
	nn_dynamics = Chain(Dense(2, 16, tanh), Dense(16, 16, tanh), Dense(16, 2))
	ps_init, st = Lux.setup(rng, nn_dynamics)
	ps_init = ComponentArray(ps_init)
end

# ╔═╡ f6a4732f-17c4-43cb-a4db-e372d33cc740
function neural_ode_func(u, p, t)
	û, _ = nn_dynamics(u, p, st)
	return û
end

# ╔═╡ 14207813-8c4e-4e8e-8870-68fd565940de
prob_neural = ODEProblem(neural_ode_func, u0_true, tspan)

# ╔═╡ 3b9da8e0-a86d-4c6d-b99b-bf996b846b57
md"""
## 3. Loss function and training loop

The loss re-solves the neural ODE with the current parameters and compares against the data. `SciMLSensitivity` provides the adjoint methods that make `solve` differentiable with respect to `p`, so `Optimization.jl` can minimize the loss with a standard gradient-based optimizer.
"""

# ╔═╡ 2535d1ef-3317-4c80-8831-9ebcf17c58e0
function predict_neural_ode(p)
	solve(prob_neural, Tsit5(); p=p, saveat=tsteps,
		sensealg=InterpolatingAdjoint(; autojacvec=ZygoteVJP())) |> Array
end

# ╔═╡ 9e9d705e-606d-48fa-8492-1c2c698ed648
function loss_neural_ode(p, _)
	pred = predict_neural_ode(p)
	return sum(abs2, ode_data .- pred)
end

# ╔═╡ b2b0f32b-a4d3-4a49-b0f2-a766ae5ed6c0
begin
	adtype = Optimization.AutoZygote()
	optf = OptimizationFunction(loss_neural_ode, adtype)
	optprob = OptimizationProblem(optf, ps_init)

	losses = Float64[]
	callback = function (state, loss)
		push!(losses, loss)
		return false
	end

	res1 = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.05); callback=callback, maxiters=300)
end

# ╔═╡ 9ecc43e1-6c72-48ce-9e8d-0fb5b19d7ece
plot(losses; xlabel="iteration", ylabel="loss", yscale=:log10, label="training loss", title="Neural ODE training")

# ╔═╡ bab00663-4b6b-45be-8162-1fcb19273082
md"""
## 4. Compare the learned dynamics to the ground truth
"""

# ╔═╡ 057ddff9-dbe8-497a-988f-af92ac76e1b6
begin
	fitted = predict_neural_ode(res1.u)
	plot(tsteps, ode_data[1, :]; label="data (u₁)", lw=2, xlabel="t")
	plot!(tsteps, fitted[1, :]; label="neural ODE (u₁)", lw=2, ls=:dash)
end

# ╔═╡ dc2a7300-5f59-4927-ad6d-da3f32171f3d
md"""
## Takeaways

* A neural ODE is just an `ODEProblem` whose vector field is a neural network — everything else (solvers, adaptive stepping, event handling) comes for free from the SciML solver stack.
* `SciMLSensitivity.jl` supplies the adjoint sensitivity methods that make `solve` compatible with reverse-mode autodiff (`Zygote`), which is what lets `Optimization.jl` train it like any other model.
* This idea generalizes directly to *partially* known physics — see the next notebook on Universal Differential Equations.

Next: [`03_universal_differential_equations.jl`](./03_universal_differential_equations.jl).
"""

# ╔═╡ Cell order:
# ╟─15b5f862-02fe-42eb-8c6f-edf193d60a79
# ╠═cd436a19-abad-49c8-99ea-9bea52c61ce4
# ╟─b217208c-c9b2-431b-942b-3f9f4494c302
# ╠═da941467-3549-4cfd-9a88-1590735978fe
# ╠═2795b34b-5dfd-4037-b1cf-c008739f7540
# ╟─b85ddfb2-4919-4982-8b02-9af675ebeed7
# ╠═f7eeda23-9a2a-417f-bcc1-5751c0e27cbb
# ╠═f6a4732f-17c4-43cb-a4db-e372d33cc740
# ╠═14207813-8c4e-4e8e-8870-68fd565940de
# ╟─3b9da8e0-a86d-4c6d-b99b-bf996b846b57
# ╠═2535d1ef-3317-4c80-8831-9ebcf17c58e0
# ╠═9e9d705e-606d-48fa-8492-1c2c698ed648
# ╠═b2b0f32b-a4d3-4a49-b0f2-a766ae5ed6c0
# ╠═9ecc43e1-6c72-48ce-9e8d-0fb5b19d7ece
# ╟─bab00663-4b6b-45be-8162-1fcb19273082
# ╠═057ddff9-dbe8-497a-988f-af92ac76e1b6
# ╟─dc2a7300-5f59-4927-ad6d-da3f32171f3d
