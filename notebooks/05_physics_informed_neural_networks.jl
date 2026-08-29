### A Pluto.jl notebook ###
# 0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ 96b58598-c72c-4d06-9ecd-75609c3168c5
md"""
# 5 · Physics-Informed Neural Networks (PINNs)

Instead of discretizing a PDE on a mesh, a **Physics-Informed Neural Network** trains a neural network $u_\theta(x)$ to satisfy the PDE's residual (and its boundary conditions) at randomly sampled collocation points. `NeuralPDE.jl` turns a symbolic PDE specification (written with `ModelingToolkit.jl`) directly into an `Optimization.jl` training problem.

We solve the 1D Poisson equation

$$u''(x) = f(x) = -\pi^2 \sin(\pi x), \qquad u(0) = 0,\ u(1) = 0$$

whose exact solution is $u(x) = \sin(\pi x)$, so we can check the network's accuracy directly.
"""

# ╔═╡ 7f4e7484-1454-4ad1-bcdb-d6f3bfc6289e
using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimisers, Random, Plots
import ModelingToolkit: Interval

# ╔═╡ fa796fbb-0f9c-42e0-9ea1-2c93b2ccbdd7
md"""
## 1. Symbolic problem specification

`ModelingToolkit.jl` lets us write down the PDE almost exactly as it looks on paper.
"""

# ╔═╡ 542952c6-2d8f-4848-8f81-d21cbf429b03
begin
	ModelingToolkit.@parameters x
	ModelingToolkit.@variables u(..)
	Dxx = ModelingToolkit.Differential(x)^2

	f(x) = -π^2 * sin(π * x)
	eq = Dxx(u(x)) ~ f(x)

	bcs = [u(0.0) ~ 0.0, u(1.0) ~ 0.0]
	domains = [x ∈ Interval(0.0, 1.0)]
end

# ╔═╡ 00fa8b07-7999-4229-9cfc-60c0188d38f6
md"""
## 2. The neural network ansatz

A small `Lux` network stands in for $u_\theta(x)$. `NeuralPDE.jl` handles differentiating it (via automatic differentiation) to build the residual loss automatically.
"""

# ╔═╡ 9b44b837-bb1e-4b95-93d6-5f31302c75b6
begin
	rng5 = Random.default_rng()
	Random.seed!(rng5, 0)
	chain = Chain(Dense(1, 16, tanh), Dense(16, 16, tanh), Dense(16, 1))
end

# ╔═╡ a53c9328-0158-49a9-b6a1-f7d5d75ef802
discretization = PhysicsInformedNN(chain, QuadratureTraining())

# ╔═╡ 838921c6-f193-4d52-b313-68e09ae5effc
begin
	@named pde_system = PDESystem(eq, bcs, domains, [x], [u(x)])
	prob_pinn = discretize(pde_system, discretization)
end

# ╔═╡ d8dfe9ba-9462-4c75-8906-48aec5cdaa76
md"""
## 3. Train the network to minimize the PDE + boundary-condition residual
"""

# ╔═╡ 7cf53913-0bac-45fe-bf20-066412ebc687
begin
	pinn_losses = Float64[]
	pinn_callback = function (state, loss)
		push!(pinn_losses, loss)
		return false
	end

	res_pinn = Optimization.solve(prob_pinn, OptimizationOptimisers.Adam(0.01); callback=pinn_callback, maxiters=1500)
end

# ╔═╡ 1c8ef438-a8c9-4022-af2f-1d0cec577c2e
plot(pinn_losses; xlabel="iteration", ylabel="PDE + BC residual loss", yscale=:log10, label="training loss")

# ╔═╡ c98ee405-0b04-48fd-94b1-880cd1806299
md"""
## 4. Compare against the exact solution
"""

# ╔═╡ a60a403a-6c13-481d-b09a-0a88f71035b7
begin
	phi = discretization.phi
	final_ps = res_pinn.u

	xs = 0.0:0.01:1.0
	u_predicted = [first(phi([xi], final_ps)) for xi in xs]
	u_exact = sin.(π .* xs)

	plot(xs, u_exact; label="exact  u(x) = sin(πx)", lw=2, xlabel="x", ylabel="u")
	plot!(xs, u_predicted; label="PINN", lw=2, ls=:dash)
end

# ╔═╡ a2af7294-1370-45ea-aec3-21c2e4501d45
maximum(abs.(u_predicted .- u_exact))

# ╔═╡ 57ff8c53-65cb-45ab-8294-3494b6f8c416
md"""
## Takeaways

* A PINN needs no mesh: the loss is just "how well does the network satisfy the equation and boundary conditions," evaluated at sampled points.
* `ModelingToolkit.jl` symbolic expressions let `NeuralPDE.jl` build the residual and its derivatives automatically via automatic differentiation, rather than requiring you to hand-derive them.
* The same recipe scales to higher-dimensional PDEs and systems of PDEs where classical mesh-based methods become expensive — the main change is a bigger network and more training time, not new mathematics in the notebook.
"""

# ╔═╡ Cell order:
# ╟─96b58598-c72c-4d06-9ecd-75609c3168c5
# ╠═7f4e7484-1454-4ad1-bcdb-d6f3bfc6289e
# ╟─fa796fbb-0f9c-42e0-9ea1-2c93b2ccbdd7
# ╠═542952c6-2d8f-4848-8f81-d21cbf429b03
# ╟─00fa8b07-7999-4229-9cfc-60c0188d38f6
# ╠═9b44b837-bb1e-4b95-93d6-5f31302c75b6
# ╠═a53c9328-0158-49a9-b6a1-f7d5d75ef802
# ╠═838921c6-f193-4d52-b313-68e09ae5effc
# ╟─d8dfe9ba-9462-4c75-8906-48aec5cdaa76
# ╠═7cf53913-0bac-45fe-bf20-066412ebc687
# ╠═1c8ef438-a8c9-4022-af2f-1d0cec577c2e
# ╟─c98ee405-0b04-48fd-94b1-880cd1806299
# ╠═a60a403a-6c13-481d-b09a-0a88f71035b7
# ╠═a2af7294-1370-45ea-aec3-21c2e4501d45
# ╟─57ff8c53-65cb-45ab-8294-3494b6f8c416
