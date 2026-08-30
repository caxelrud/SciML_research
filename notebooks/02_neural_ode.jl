### A Pluto.jl notebook ###
# 0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ eb709ba6-0571-433d-a9a7-2188f27429cc
md"""
# 2 · Neural ODEs

A **Neural ODE** replaces the right-hand side of a differential equation with a neural network:

$$\dot u = f_\theta(u, t)$$

where $f_\theta$ is a small neural network with trainable parameters $\theta$. Training means: simulate the ODE forward, compare to data, and backpropagate *through the ODE solver* to update $\theta$. This notebook trains a neural ODE to reproduce a 2D spiral trajectory, following the classic example from Chen et al. (2018), *Neural Ordinary Differential Equations*.
"""

# ╔═╡ fda6fc57-a084-4534-87f4-b4fa7815abce
using Lux, OrdinaryDiffEq, SciMLSensitivity, Optimization, OptimizationOptimisers, ComponentArrays, Random, Plots

# ╔═╡ 38f9aeb1-5eda-45fa-b0cf-55f71b7dcd5a
md"""
## 1. Generate synthetic "ground truth" data

We pretend we don't know the true dynamics and only observe noiseless samples of a 2D spiral produced by a simple linear ODE $\dot u = Au$.
"""

# ╔═╡ a308c01a-ad6b-4c0f-bf8f-32636a4714f3
begin
	rng = Random.default_rng()
	Random.seed!(rng, 1)

	true_A = Float32[-0.1 2.0; -2.0 -0.1]
	true_odefunc(u, p, t) = true_A * (u .^ 3)   # nonlinear spiral (cubic term)

	u0_true = Float32[2.0, 0.0]
	tspan = (0.0f0, 1.5f0)
	tsteps = range(tspan[1], tspan[2]; length=40)

	true_prob = ODEProblem(true_odefunc, u0_true, tspan)
	true_sol = solve(true_prob, Tsit5(); saveat=tsteps)
	ode_data = Array(true_sol)
end

# ╔═╡ 5cc3817a-f689-43e5-812f-3d0dce05da20
scatter(tsteps, ode_data[1, :]; label="u₁ (data)", xlabel="t", title="Observed spiral trajectory (state 1)")

# ╔═╡ 98399a9f-2f23-466b-adfd-6a4fc24e50c0
md"""
## 2. Define the neural ODE

Instead of the (unknown, in a real problem) true dynamics, we parameterize $f_\theta$ with a small `Lux` multilayer perceptron and wrap it in an `ODEProblem`.
"""

# ╔═╡ 58eb339d-33ff-4035-be77-54d1ceb6321e
begin
	nn_dynamics = Chain(Dense(2, 16, tanh), Dense(16, 16, tanh), Dense(16, 2))
	ps_init, st = Lux.setup(rng, nn_dynamics)
	ps_init = ComponentArray(ps_init)
end

# ╔═╡ 392afa83-72c2-4467-b433-f4dc78b1134b
function neural_ode_func(u, p, t)
	û, _ = nn_dynamics(u, p, st)
	return û
end

# ╔═╡ 4870cefb-2c73-4533-9d24-8f31c2361095
prob_neural = ODEProblem(neural_ode_func, u0_true, tspan)

# ╔═╡ 489839c5-f017-4568-a93d-51c59df213b5
md"""
## 3. Loss function and training loop

The loss re-solves the neural ODE with the current parameters and compares against the data. `SciMLSensitivity` provides the adjoint methods that make `solve` differentiable with respect to `p`, so `Optimization.jl` can minimize the loss with a standard gradient-based optimizer.
"""

# ╔═╡ 75c97823-d6e2-401b-80de-1d8260cf044b
function predict_neural_ode(p)
	solve(prob_neural, Tsit5(); p=p, saveat=tsteps,
		sensealg=InterpolatingAdjoint(; autojacvec=ZygoteVJP())) |> Array
end

# ╔═╡ f0370f28-a75b-4089-a740-e25c97ba02d3
function loss_neural_ode(p, _)
	pred = predict_neural_ode(p)
	return sum(abs2, ode_data .- pred)
end

# ╔═╡ d93010c4-0b9a-4ea3-8f49-8df9b5e6b111
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

# ╔═╡ e3efc67b-e078-4c03-bb06-c1bc3aa64cf7
plot(losses; xlabel="iteration", ylabel="loss", yscale=:log10, label="training loss", title="Neural ODE training")

# ╔═╡ da31f11c-8926-4164-b997-ce89a25e7a9a
md"""
## 4. Compare the learned dynamics to the ground truth
"""

# ╔═╡ c10ee6f0-cb26-4d89-a0eb-24a7c44d4933
begin
	fitted = predict_neural_ode(res1.u)
	plot(tsteps, ode_data[1, :]; label="data (u₁)", lw=2, xlabel="t")
	plot!(tsteps, fitted[1, :]; label="neural ODE (u₁)", lw=2, ls=:dash)
end

# ╔═╡ da65432e-d910-471a-9172-6c81b7c394be
md"""
## Takeaways

* A neural ODE is just an `ODEProblem` whose vector field is a neural network — everything else (solvers, adaptive stepping, event handling) comes for free from the SciML solver stack.
* `SciMLSensitivity.jl` supplies the adjoint sensitivity methods that make `solve` compatible with reverse-mode autodiff (`Zygote`), which is what lets `Optimization.jl` train it like any other model.
* This idea generalizes directly to *partially* known physics — see the next notebook on Universal Differential Equations.

Next: [`03_universal_differential_equations.jl`](./03_universal_differential_equations.jl).
"""

# ╔═╡ Cell order:
# ╟─eb709ba6-0571-433d-a9a7-2188f27429cc
# ╠═fda6fc57-a084-4534-87f4-b4fa7815abce
# ╟─38f9aeb1-5eda-45fa-b0cf-55f71b7dcd5a
# ╠═a308c01a-ad6b-4c0f-bf8f-32636a4714f3
# ╠═5cc3817a-f689-43e5-812f-3d0dce05da20
# ╟─98399a9f-2f23-466b-adfd-6a4fc24e50c0
# ╠═58eb339d-33ff-4035-be77-54d1ceb6321e
# ╠═392afa83-72c2-4467-b433-f4dc78b1134b
# ╠═4870cefb-2c73-4533-9d24-8f31c2361095
# ╟─489839c5-f017-4568-a93d-51c59df213b5
# ╠═75c97823-d6e2-401b-80de-1d8260cf044b
# ╠═f0370f28-a75b-4089-a740-e25c97ba02d3
# ╠═d93010c4-0b9a-4ea3-8f49-8df9b5e6b111
# ╠═e3efc67b-e078-4c03-bb06-c1bc3aa64cf7
# ╟─da31f11c-8926-4164-b997-ce89a25e7a9a
# ╠═c10ee6f0-cb26-4d89-a0eb-24a7c44d4933
# ╟─da65432e-d910-471a-9172-6c81b7c394be
