### A Pluto.jl notebook ###
# 0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ 5f924d3f-a2f4-4238-9f97-8fd075405c5a
md"""
# O&G · 3 — Universal Differential Equations: material balance with unknown aquifer support

A reservoir tank material-balance model tracks average reservoir pressure $P$ as fluid is withdrawn:

$$\frac{dP}{dt} = -c_1\, q(t) + (\text{pressure support})$$

The depletion term $-c_1 q(t)$ (rate times a known compressibility/pore-volume factor) is usually well understood. What's often *not* well characterized is aquifer influx or injection support — its strength and functional form are exactly the kind of thing reservoir engineers argue about. We keep the known depletion term as an equation and let a neural network learn the unknown support term directly from the pressure history, following the Universal Differential Equations recipe of [`03_universal_differential_equations.jl`](../03_universal_differential_equations.jl).
"""

# ╔═╡ 32ec131c-2768-485b-9d8d-f92e6c878d39
using Lux, OrdinaryDiffEq, SciMLSensitivity, Optimization, OptimizationOptimisers, ComponentArrays, Random, Plots

# ╔═╡ 1679d97d-21ad-4c14-8e47-2f2468ad4a70
md"""
## 1. Ground truth: depletion plus a Fetkovich-style aquifer

We generate synthetic pressure data from a full model that *does* include an aquifer influx term $J(P_i - P)$ (a common approximation for a weak, tank-like aquifer) — but this term is what we'll later hide from the identified model.
"""

# ╔═╡ a1df44ec-ef02-428d-92f6-6b65dd6bd7ae
production_rate(t) = t < 2.0 ? 500.0 : 800.0   # a second well comes online at t = 2 yr

# ╔═╡ 83edd5d7-0707-4e1a-b5f3-561c853cdc18
function tank_true!(du, u, p, t)
	P = u[1]
	c1, J, Pi = p
	q = production_rate(t)
	du[1] = -c1 * q + J * (Pi - P)
	return nothing
end

# ╔═╡ 63071812-e1e9-4913-b1fa-fbac93c7776d
begin
	rng_og3 = Random.default_rng()
	Random.seed!(rng_og3, 21)

	Pi_true = 4500.0            # psi, initial reservoir pressure
	c1_true = 0.02
	J_true = 0.015
	u0_tank = [Pi_true]
	tspan_tank = (0.0, 6.0)
	tsteps_tank = range(tspan_tank[1], tspan_tank[2]; length=48)

	tank_prob = ODEProblem(tank_true!, u0_tank, tspan_tank, (c1_true, J_true, Pi_true))
	tank_sol = solve(tank_prob, Tsit5(); saveat=tsteps_tank, tstops=[2.0])
	pressure_data = Array(tank_sol) .+ 8.0 .* randn(rng_og3, 1, length(tsteps_tank))
end

# ╔═╡ 3c174edc-266d-4681-9dea-e978136915ff
scatter(tsteps_tank, pressure_data[1, :]; label="measured reservoir pressure", xlabel="time (yr)", ylabel="P (psi)")

# ╔═╡ 55683bda-10ce-4dd9-a2b3-bd2c1a7c856c
md"""
## 2. The hybrid model

We keep the depletion term $-c_1 q(t)$ exactly (assume $c_1$ is known from PVT and pore-volume data), and replace the aquifer-support term with a small neural network $U_\theta(P)$.
"""

# ╔═╡ 4264edcb-862b-4078-bc48-76e8c0621f0b
begin
	support_nn = Chain(Dense(1, 8, tanh), Dense(8, 8, tanh), Dense(8, 1))
	support_ps_init, support_st = Lux.setup(rng_og3, support_nn)
	support_ps_init = ComponentArray(support_ps_init)
end

# ╔═╡ d5e37937-1fea-4d60-9cac-14c27f44f7c7
c1_known = 0.02

# ╔═╡ c2876a27-7e2d-4ec0-b701-6b1a51e787cd
function tank_ude!(du, u, p, t)
	P = u[1]
	q = production_rate(t)
	support, _ = support_nn([P / 1000.0], p, support_st)
	du[1] = -c1_known * q + 1000.0 * support[1]
	return nothing
end

# ╔═╡ 20bcda9a-a1d5-4e6a-9cef-05ffee9c6502
tank_ude_prob = ODEProblem(tank_ude!, u0_tank, tspan_tank)

# ╔═╡ 65e00e6a-cbfb-4f04-91b5-4f5197bbecfb
md"""
## 3. Train the neural support term against the pressure history
"""

# ╔═╡ 78e96c5e-d50d-43ba-9eab-c1fdf5f2ad72
function predict_tank(p)
	solve(tank_ude_prob, Tsit5(); p=p, saveat=tsteps_tank, tstops=[2.0],
		sensealg=InterpolatingAdjoint(; autojacvec=ZygoteVJP())) |> Array
end

# ╔═╡ c1228f6b-ac2b-4610-ad12-19256d83eca6
function loss_tank(p, _)
	pred = predict_tank(p)
	size(pred) == size(pressure_data) || return Inf
	return sum(abs2, pressure_data .- pred)
end

# ╔═╡ 5bb24598-5dbc-4efb-94b5-c8280c9d9a2f
begin
	optf_og3 = OptimizationFunction(loss_tank, Optimization.AutoZygote())
	optprob_og3 = OptimizationProblem(optf_og3, support_ps_init)

	tank_losses = Float64[]
	tank_callback = function (state, loss)
		push!(tank_losses, loss)
		return false
	end

	res_tank = Optimization.solve(optprob_og3, OptimizationOptimisers.Adam(0.03); callback=tank_callback, maxiters=600)
end

# ╔═╡ 907a880d-6494-4216-8132-b7191ab3f44e
plot(tank_losses; xlabel="iteration", ylabel="loss", yscale=:log10, label="UDE training loss")

# ╔═╡ a2c55c8f-a886-45ad-ad94-22d3992c2d3f
md"""
## 4. Fit quality and the recovered support curve
"""

# ╔═╡ 6dc0106a-da70-4bd6-917f-db353ec0071f
begin
	tank_fit = predict_tank(res_tank.u)
	scatter(tsteps_tank, pressure_data[1, :]; label="data", xlabel="time (yr)", ylabel="P (psi)")
	plot!(tsteps_tank, tank_fit[1, :]; label="UDE model", lw=2)
end

# ╔═╡ c23809a6-1d9f-4375-92f4-16f016d498fa
begin
	p_grid = 3800.0:10.0:4500.0
	learned_support = [1000.0 * first(support_nn([p / 1000.0], res_tank.u, support_st)[1]) for p in p_grid]
	true_support = [J_true * (Pi_true - p) for p in p_grid]

	plot(p_grid, true_support; label="true aquifer support  J(Pi - P)", lw=2, xlabel="P (psi)", ylabel="dP/dt contribution")
	plot!(p_grid, learned_support; label="learned U_θ(P)", lw=2, ls=:dash)
end

# ╔═╡ 19b4bc9c-1ad8-48ef-8d74-1e220670f7b5
md"""
## Takeaways

* Only the aquifer term — the part reservoir engineers are genuinely uncertain about — was replaced by a network; the depletion physics stayed a normal equation.
* Because `production_rate(t)` has a jump at $t=2$, passing `tstops=[2.0]` to `solve` keeps the adaptive integrator accurate across the discontinuity instead of stepping over it.
* The recovered $U_\theta(P)$ curve is itself useful: comparing its shape to the true influx law is exactly how an engineer would sanity-check whether the identified support behaves like a real aquifer (roughly linear in drawdown) or something else entirely (e.g. injection support, which would look different).
"""

# ╔═╡ Cell order:
# ╟─5f924d3f-a2f4-4238-9f97-8fd075405c5a
# ╠═32ec131c-2768-485b-9d8d-f92e6c878d39
# ╟─1679d97d-21ad-4c14-8e47-2f2468ad4a70
# ╠═a1df44ec-ef02-428d-92f6-6b65dd6bd7ae
# ╠═83edd5d7-0707-4e1a-b5f3-561c853cdc18
# ╠═63071812-e1e9-4913-b1fa-fbac93c7776d
# ╠═3c174edc-266d-4681-9dea-e978136915ff
# ╟─55683bda-10ce-4dd9-a2b3-bd2c1a7c856c
# ╠═4264edcb-862b-4078-bc48-76e8c0621f0b
# ╠═d5e37937-1fea-4d60-9cac-14c27f44f7c7
# ╠═c2876a27-7e2d-4ec0-b701-6b1a51e787cd
# ╠═20bcda9a-a1d5-4e6a-9cef-05ffee9c6502
# ╟─65e00e6a-cbfb-4f04-91b5-4f5197bbecfb
# ╠═78e96c5e-d50d-43ba-9eab-c1fdf5f2ad72
# ╠═c1228f6b-ac2b-4610-ad12-19256d83eca6
# ╠═5bb24598-5dbc-4efb-94b5-c8280c9d9a2f
# ╠═907a880d-6494-4216-8132-b7191ab3f44e
# ╟─a2c55c8f-a886-45ad-ad94-22d3992c2d3f
# ╠═6dc0106a-da70-4bd6-917f-db353ec0071f
# ╠═c23809a6-1d9f-4375-92f4-16f016d498fa
# ╟─19b4bc9c-1ad8-48ef-8d74-1e220670f7b5
