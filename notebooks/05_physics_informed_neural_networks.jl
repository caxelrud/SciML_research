### A Pluto.jl notebook ###
# 0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ 602031da-e3e9-47f4-850a-b47fb0eafd42
md"""
# 5 · Physics-Informed Neural Networks (PINNs)

Instead of discretizing a PDE on a mesh, a **Physics-Informed Neural Network** trains a neural network $u_\theta(x)$ to satisfy the PDE's residual (and its boundary conditions) at randomly sampled collocation points. `NeuralPDE.jl` turns a symbolic PDE specification (written with `ModelingToolkit.jl`) directly into an `Optimization.jl` training problem.

We solve the 1D Poisson equation

$$u''(x) = f(x) = -\pi^2 \sin(\pi x), \qquad u(0) = 0,\ u(1) = 0$$

whose exact solution is $u(x) = \sin(\pi x)$, so we can check the network's accuracy directly.
"""

# ╔═╡ 1bdf4c6f-bf02-499e-bda3-5d5903b178b7
using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimisers, DomainSets, Random, Plots

# ╔═╡ 6fb9f829-8b08-4d4a-aaa7-9c9cb9bfe3d1
md"""
## 1. Symbolic problem specification

`ModelingToolkit.jl` lets us write down the PDE almost exactly as it looks on paper.
"""

# ╔═╡ 7c7da7f9-597e-4f20-b3f4-892ccb531224
ModelingToolkit.@parameters x

# ╔═╡ 3dc43828-3c12-4c2a-abea-034386a8ab2d
ModelingToolkit.@variables u(..)

# ╔═╡ 33cfdc32-768a-41ed-815d-4ab24ab2d994
begin
	Dxx = ModelingToolkit.Differential(x)^2

	f(x) = -π^2 * sin(π * x)
	eq = Dxx(u(x)) ~ f(x)

	bcs = [u(0.0) ~ 0.0, u(1.0) ~ 0.0]
	domains = [x ∈ Interval(0.0, 1.0)]
end

# ╔═╡ e75ff946-130b-4d35-bbe2-6ac2f2b63be4
md"""
## 2. The neural network ansatz

A small `Lux` network stands in for $u_\theta(x)$. `NeuralPDE.jl` handles differentiating it (via automatic differentiation) to build the residual loss automatically.
"""

# ╔═╡ e3d4c68e-b649-4467-9f34-8f2a87890bf4
begin
	rng5 = Random.default_rng()
	Random.seed!(rng5, 0)
	chain = Chain(Dense(1, 16, tanh), Dense(16, 16, tanh), Dense(16, 1))
end

# ╔═╡ 43866827-f566-4867-8375-bafec3cda7c0
discretization = PhysicsInformedNN(chain, QuadratureTraining())

# ╔═╡ f677fdbf-3285-4324-80a1-420322076ce3
begin
	@named pde_system = PDESystem(eq, bcs, domains, [x], [u(x)])
	prob_pinn = discretize(pde_system, discretization)
end

# ╔═╡ 88f24471-3a44-40cf-add2-94a5f1b411f9
md"""
## 3. Train the network to minimize the PDE + boundary-condition residual
"""

# ╔═╡ f86bc474-77c9-45a1-8376-1f3de972a370
begin
	pinn_losses = Float64[]
	pinn_callback = function (state, loss)
		push!(pinn_losses, loss)
		return false
	end

	res_pinn = Optimization.solve(prob_pinn, OptimizationOptimisers.Adam(0.01); callback=pinn_callback, maxiters=1500)
end

# ╔═╡ b99f8520-32f1-4131-919c-5363073afd75
plot(pinn_losses; xlabel="iteration", ylabel="PDE + BC residual loss", yscale=:log10, label="training loss")

# ╔═╡ 52a16944-1828-4e98-b63a-128ab0030967
md"""
## 4. Compare against the exact solution
"""

# ╔═╡ 0ffe6510-5a60-4fc1-b5d6-994e9025ee94
begin
	phi = discretization.phi
	final_ps = res_pinn.u

	xs = 0.0:0.01:1.0
	u_predicted = [first(phi([xi], final_ps)) for xi in xs]
	u_exact = sin.(π .* xs)

	plot(xs, u_exact; label="exact  u(x) = sin(πx)", lw=2, xlabel="x", ylabel="u")
	plot!(xs, u_predicted; label="PINN", lw=2, ls=:dash)
end

# ╔═╡ 4a5a62ff-aaff-4071-b7ee-14f70559398c
maximum(abs.(u_predicted .- u_exact))

# ╔═╡ 31d9df5f-b568-4fc4-b3b0-202e10462e89
md"""
## Takeaways

* A PINN needs no mesh: the loss is just "how well does the network satisfy the equation and boundary conditions," evaluated at sampled points.
* `ModelingToolkit.jl` symbolic expressions let `NeuralPDE.jl` build the residual and its derivatives automatically via automatic differentiation, rather than requiring you to hand-derive them.
* The same recipe scales to higher-dimensional PDEs and systems of PDEs where classical mesh-based methods become expensive — the main change is a bigger network and more training time, not new mathematics in the notebook.
"""

# ╔═╡ Cell order:
# ╟─602031da-e3e9-47f4-850a-b47fb0eafd42
# ╠═1bdf4c6f-bf02-499e-bda3-5d5903b178b7
# ╟─6fb9f829-8b08-4d4a-aaa7-9c9cb9bfe3d1
# ╠═7c7da7f9-597e-4f20-b3f4-892ccb531224
# ╠═3dc43828-3c12-4c2a-abea-034386a8ab2d
# ╠═33cfdc32-768a-41ed-815d-4ab24ab2d994
# ╟─e75ff946-130b-4d35-bbe2-6ac2f2b63be4
# ╠═e3d4c68e-b649-4467-9f34-8f2a87890bf4
# ╠═43866827-f566-4867-8375-bafec3cda7c0
# ╠═f677fdbf-3285-4324-80a1-420322076ce3
# ╟─88f24471-3a44-40cf-add2-94a5f1b411f9
# ╠═f86bc474-77c9-45a1-8376-1f3de972a370
# ╠═b99f8520-32f1-4131-919c-5363073afd75
# ╟─52a16944-1828-4e98-b63a-128ab0030967
# ╠═0ffe6510-5a60-4fc1-b5d6-994e9025ee94
# ╠═4a5a62ff-aaff-4071-b7ee-14f70559398c
# ╟─31d9df5f-b568-4fc4-b3b0-202e10462e89
