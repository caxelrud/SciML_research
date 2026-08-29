### A Pluto.jl notebook ###
# 0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ dce431f0-47d1-45cb-9df5-62a1ab3cb7c8
md"""
# O&G · 5 — A physics-informed neural network for pressure-transient analysis

[`04_well_test_history_matching.jl`](./04_well_test_history_matching.jl) used a closed-form *approximation* to the diffusivity equation. Here we solve the actual dimensionless radial diffusivity PDE directly with a physics-informed neural network, the same technique as [`05_physics_informed_neural_networks.jl`](../05_physics_informed_neural_networks.jl), applied to a bounded (closed outer-boundary) reservoir:

$$\frac{\partial p_D}{\partial t_D} = \frac{1}{r_D}\frac{\partial}{\partial r_D}\!\left(r_D \frac{\partial p_D}{\partial r_D}\right), \qquad r_D \in [1, r_{eD}]$$

with a constant-rate inner boundary (Neumann), a no-flow outer boundary (Neumann), and zero initial pressure depletion:

$$\left.\frac{\partial p_D}{\partial r_D}\right|_{r_D=1} = -1, \qquad \left.\frac{\partial p_D}{\partial r_D}\right|_{r_D=r_{eD}} = 0, \qquad p_D(r_D, 0) = 0$$

$p_D(1, t_D)$ — the dimensionless pressure at the wellbore — is exactly the quantity a pressure gauge records during a well test.
"""

# ╔═╡ 47eec489-d4da-4c39-a998-5d116be4f0d7
using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimisers, Random, Plots
import ModelingToolkit: Interval

# ╔═╡ 7e6fdc82-2a70-436a-980e-f877556b13b2
md"""
## 1. Symbolic PDE specification
"""

# ╔═╡ 3a64c062-926e-4071-9eaa-8d97c709ce27
begin
	reD = 50.0     # dimensionless drainage radius (re / rw)
	tDmax = 200.0

	ModelingToolkit.@parameters r t
	ModelingToolkit.@variables p(..)
	Dt = ModelingToolkit.Differential(t)
	Dr = ModelingToolkit.Differential(r)
	Drr = Dr^2

	eq = Dt(p(r, t)) ~ Drr(p(r, t)) + (1 / r) * Dr(p(r, t))

	bcs = [p(r, 0.0) ~ 0.0,
		Dr(p(1.0, t)) ~ -1.0,
		Dr(p(reD, t)) ~ 0.0]

	domains = [r ∈ Interval(1.0, reD), t ∈ Interval(0.0, tDmax)]
end

# ╔═╡ a85f2034-cec0-4b12-8ab2-eb7060ab6356
md"""
## 2. The network ansatz and training problem
"""

# ╔═╡ 1178190f-8550-44ac-8983-fe666ae447ae
begin
	rng_og5 = Random.default_rng()
	Random.seed!(rng_og5, 2)
	pde_chain = Chain(Dense(2, 24, tanh), Dense(24, 24, tanh), Dense(24, 1))
end

# ╔═╡ 0bd742ae-b6b5-4c7a-8894-f9c0f3215b8a
pde_discretization = PhysicsInformedNN(pde_chain, QuadratureTraining())

# ╔═╡ e7f05782-6740-43e6-a218-ab58f1f2d1db
begin
	@named pressure_pde_system = PDESystem(eq, bcs, domains, [r, t], [p(r, t)])
	prob_pde = discretize(pressure_pde_system, pde_discretization)
end

# ╔═╡ 2fb4515f-dae3-43d6-8a1b-7c9f77cca7c4
md"""
## 3. Train
"""

# ╔═╡ 1a9a60a9-8cdd-4380-b099-9192591af448
begin
	pde_losses = Float64[]
	pde_callback = function (state, loss)
		push!(pde_losses, loss)
		return false
	end

	res_pde = Optimization.solve(prob_pde, OptimizationOptimisers.Adam(0.01); callback=pde_callback, maxiters=2000)
end

# ╔═╡ 246bfb06-c0c9-43df-b293-835ea8cb3147
plot(pde_losses; xlabel="iteration", ylabel="PDE + BC residual", yscale=:log10, label="training loss")

# ╔═╡ 3aef3500-264d-4886-ad34-7012432b8a96
md"""
## 4. The wellbore pressure response — and a sanity check against the early-time analytical solution

Before the pressure disturbance reaches the outer boundary, the well behaves as if the reservoir were infinite-acting, and $p_D(1, t_D)$ should follow the well-known log approximation

$$p_D(1, t_D) \approx \tfrac{1}{2}\bigl[\ln(t_D) + 0.80907\bigr]$$

This gives us an independent check on the trained network that has nothing to do with the training loss itself.
"""

# ╔═╡ 5cad6a0a-a5f0-4950-9aa0-ee02d600e034
begin
	phi_pde = pde_discretization.phi
	final_pde_ps = res_pde.u

	t_grid = 1.0:1.0:150.0
	pD_wellbore = [first(phi_pde([1.0, ti], final_pde_ps)) for ti in t_grid]
	pD_analytical = [0.5 * (log(ti) + 0.80907) for ti in t_grid]

	plot(t_grid, pD_analytical; xscale=:log10, label="infinite-acting log approximation", lw=2, xlabel="tD", ylabel="pD(1, tD)")
	plot!(t_grid, pD_wellbore; xscale=:log10, label="PINN", lw=2, ls=:dash,
		title="Wellbore pressure response (PINN vs. analytical early-time solution)")
end

# ╔═╡ 67b29118-43d3-4c72-ad61-f149e48dd17e
md"""
## Takeaways

* The Neumann inner-boundary condition — the actual physical statement "the well produces at a constant rate" — is passed to `NeuralPDE.jl` as literally $\partial p_D/\partial r_D|_{r_D=1} = -1$, no discretized flux calculation required.
* At early times (before the pressure transient reaches $r_{eD}$), the PINN's wellbore response should track the classical infinite-acting log approximation; at late times it should flatten out as the closed outer boundary is felt — the same **infinite-acting → transition → boundary-dominated flow** signature every well test interpreter looks for.
* Because the whole reservoir domain was solved in one mesh-free network, extracting the pressure at *any* radius (not just the wellbore) is a free `phi([r, t], p)` evaluation — useful for interference-test analysis at an offset observation well.
"""

# ╔═╡ Cell order:
# ╟─dce431f0-47d1-45cb-9df5-62a1ab3cb7c8
# ╠═47eec489-d4da-4c39-a998-5d116be4f0d7
# ╟─7e6fdc82-2a70-436a-980e-f877556b13b2
# ╠═3a64c062-926e-4071-9eaa-8d97c709ce27
# ╟─a85f2034-cec0-4b12-8ab2-eb7060ab6356
# ╠═1178190f-8550-44ac-8983-fe666ae447ae
# ╠═0bd742ae-b6b5-4c7a-8894-f9c0f3215b8a
# ╠═e7f05782-6740-43e6-a218-ab58f1f2d1db
# ╟─2fb4515f-dae3-43d6-8a1b-7c9f77cca7c4
# ╠═1a9a60a9-8cdd-4380-b099-9192591af448
# ╠═246bfb06-c0c9-43df-b293-835ea8cb3147
# ╟─3aef3500-264d-4886-ad34-7012432b8a96
# ╠═5cad6a0a-a5f0-4950-9aa0-ee02d600e034
# ╟─67b29118-43d3-4c72-ad61-f149e48dd17e
