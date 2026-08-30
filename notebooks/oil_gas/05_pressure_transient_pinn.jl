### A Pluto.jl notebook ###
# 0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ 7bf3162b-c065-48af-94ef-aa076e333ce1
md"""
# O&G · 5 — A physics-informed neural network for pressure-transient analysis

[`04_well_test_history_matching.jl`](./04_well_test_history_matching.jl) used a closed-form *approximation* to the diffusivity equation. Here we solve the actual dimensionless radial diffusivity PDE directly with a physics-informed neural network, the same technique as [`05_physics_informed_neural_networks.jl`](../05_physics_informed_neural_networks.jl), applied to a bounded (closed outer-boundary) reservoir:

$$\frac{\partial p_D}{\partial t_D} = \frac{1}{r_D}\frac{\partial}{\partial r_D}\!\left(r_D \frac{\partial p_D}{\partial r_D}\right), \qquad r_D \in [1, r_{eD}]$$

with a constant-rate inner boundary (Neumann), a no-flow outer boundary (Neumann), and zero initial pressure depletion:

$$\left.\frac{\partial p_D}{\partial r_D}\right|_{r_D=1} = -1, \qquad \left.\frac{\partial p_D}{\partial r_D}\right|_{r_D=r_{eD}} = 0, \qquad p_D(r_D, 0) = 0$$

$p_D(1, t_D)$ — the dimensionless pressure at the wellbore — is exactly the quantity a pressure gauge records during a well test.
"""

# ╔═╡ 690e4d8c-63d5-4923-8fba-6897c1b6c514
using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimisers, DomainSets, Random, Plots

# ╔═╡ ca742b59-2e32-4031-882a-8c63eb2ed935
md"""
## 1. Symbolic PDE specification
"""

# ╔═╡ da989a1d-3ee9-4324-8746-df2c62a3bb03
ModelingToolkit.@parameters r t

# ╔═╡ f8103176-67fd-46e8-90c9-106b5546b60d
ModelingToolkit.@variables p(..)

# ╔═╡ b9befe3c-08a9-48b0-9149-dfe80000e0ef
begin
	reD = 20.0     # dimensionless drainage radius (re / rw)
	tDmax = 40.0

	Dt = ModelingToolkit.Differential(t)
	Dr = ModelingToolkit.Differential(r)
	Drr = Dr^2

	eq = Dt(p(r, t)) ~ Drr(p(r, t)) + (1 / r) * Dr(p(r, t))

	bcs = [p(r, 0.0) ~ 0.0,
		Dr(p(1.0, t)) ~ -1.0,
		Dr(p(reD, t)) ~ 0.0]

	domains = [r ∈ Interval(1.0, reD), t ∈ Interval(0.0, tDmax)]
end

# ╔═╡ 46f09948-45d6-402b-b750-05e9aef73d2f
md"""
## 2. The network ansatz and training problem
"""

# ╔═╡ 3428e6f7-993f-42dd-ae42-87bfc7f78d9e
begin
	rng_og5 = Random.default_rng()
	Random.seed!(rng_og5, 2)
	pde_chain = Chain(Dense(2, 24, tanh), Dense(24, 24, tanh), Dense(24, 1))
end

# ╔═╡ 0fec7628-a0fd-4961-8e19-132587c2f3e1
pde_discretization = PhysicsInformedNN(pde_chain, GridTraining(0.25))

# ╔═╡ 19bd3f09-29a4-41e5-adc9-b8fde51e8f76
begin
	@named pressure_pde_system = PDESystem(eq, bcs, domains, [r, t], [p(r, t)])
	prob_pde = discretize(pressure_pde_system, pde_discretization)
end

# ╔═╡ 3bf732cc-e60a-4f01-91f2-f2f4de31a845
md"""
## 3. Train
"""

# ╔═╡ a68ee053-c646-4634-bde7-5c21337e84e2
begin
	pde_losses = Float64[]
	pde_callback = function (state, loss)
		push!(pde_losses, loss)
		return false
	end

	res_pde = Optimization.solve(prob_pde, OptimizationOptimisers.Adam(0.01); callback=pde_callback, maxiters=3000)
end

# ╔═╡ 73b43674-1968-4192-9cbb-8487e1cfd699
plot(pde_losses; xlabel="iteration", ylabel="PDE + BC residual", yscale=:log10, label="training loss")

# ╔═╡ 78f5200f-641c-4174-80f1-41bbb42603b2
md"""
## 4. The wellbore pressure response — and a sanity check against the early-time analytical solution

Before the pressure disturbance reaches the outer boundary, the well behaves as if the reservoir were infinite-acting, and $p_D(1, t_D)$ should follow the well-known log approximation

$$p_D(1, t_D) \approx \tfrac{1}{2}\bigl[\ln(t_D) + 0.80907\bigr]$$

This reservoir is small ($r_{eD}=20$), so pseudo-steady-state sets in early — around $t_D \approx 0.1\,r_{eD}^2 = 40$, right at the edge of the domain we trained over. Expect the PINN curve to track the infinite-acting line only at the start of the range and bend away from it as $t_D$ climbs toward that boundary-dominated regime — that bend is the reservoir feeling its outer edge, not a training artifact.
"""

# ╔═╡ 47eb7ded-395a-42a8-9b1b-6c923d2047d9
begin
	phi_pde = pde_discretization.phi
	final_pde_ps = res_pde.u

	t_grid = 1.0:1.0:30.0
	pD_wellbore = [first(phi_pde([1.0, ti], final_pde_ps)) for ti in t_grid]
	pD_analytical = [0.5 * (log(ti) + 0.80907) for ti in t_grid]

	plot(t_grid, pD_analytical; xscale=:log10, label="infinite-acting log approximation", lw=2, xlabel="tD", ylabel="pD(1, tD)")
	plot!(t_grid, pD_wellbore; xscale=:log10, label="PINN", lw=2, ls=:dash,
		title="Wellbore pressure response (PINN vs. analytical early-time solution)")
end

# ╔═╡ 385aa7ee-a1b9-42d1-a5bf-d0925c25169d
md"""
## Takeaways

* The Neumann inner-boundary condition — the actual physical statement "the well produces at a constant rate" — is passed to `NeuralPDE.jl` as literally $\partial p_D/\partial r_D|_{r_D=1} = -1$, no discretized flux calculation required.
* At early times the PINN's wellbore response tracks the classical infinite-acting log approximation; as $t_D$ approaches the pseudo-steady-state threshold ($\approx 0.1\,r_{eD}^2$) it bends away — the same **infinite-acting → transition → boundary-dominated flow** signature every well test interpreter looks for, reproduced here from the PDE alone rather than assumed.
* Because the whole reservoir domain was solved in one mesh-free network, extracting the pressure at *any* radius (not just the wellbore) is a free `phi([r, t], p)` evaluation — useful for interference-test analysis at an offset observation well.
"""

# ╔═╡ Cell order:
# ╟─7bf3162b-c065-48af-94ef-aa076e333ce1
# ╠═690e4d8c-63d5-4923-8fba-6897c1b6c514
# ╟─ca742b59-2e32-4031-882a-8c63eb2ed935
# ╠═da989a1d-3ee9-4324-8746-df2c62a3bb03
# ╠═f8103176-67fd-46e8-90c9-106b5546b60d
# ╠═b9befe3c-08a9-48b0-9149-dfe80000e0ef
# ╟─46f09948-45d6-402b-b750-05e9aef73d2f
# ╠═3428e6f7-993f-42dd-ae42-87bfc7f78d9e
# ╠═0fec7628-a0fd-4961-8e19-132587c2f3e1
# ╠═19bd3f09-29a4-41e5-adc9-b8fde51e8f76
# ╟─3bf732cc-e60a-4f01-91f2-f2f4de31a845
# ╠═a68ee053-c646-4634-bde7-5c21337e84e2
# ╠═73b43674-1968-4192-9cbb-8487e1cfd699
# ╟─78f5200f-641c-4174-80f1-41bbb42603b2
# ╠═47eb7ded-395a-42a8-9b1b-6c923d2047d9
# ╟─385aa7ee-a1b9-42d1-a5bf-d0925c25169d
