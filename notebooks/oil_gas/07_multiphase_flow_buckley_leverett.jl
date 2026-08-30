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

# ╔═╡ b7f36c98-470e-4738-9886-69364724fefe
md"""
# O&G · 7 — Multiphase flow: Buckley–Leverett waterflood displacement

Every earlier O&G notebook treated a single flowing phase. Waterflooding — injecting water to push oil toward a producer — is inherently two-phase, and its classic theory is the **Buckley–Leverett** equation. For 1D, incompressible, immiscible displacement with no capillary pressure, the water saturation $S_w$ obeys the hyperbolic conservation law

$$\frac{\partial S_w}{\partial t} + v \, f_w'(S_w) \, \frac{\partial S_w}{\partial x} = 0$$

where $f_w(S_w)$ is the **fractional flow** of water (the fraction of total flow that is water at a given saturation) and $v$ is the interstitial velocity. Because $f_w$ is S-shaped, naively following each saturation's characteristic produces a triple-valued, unphysical profile — resolved by **Welge's tangent construction**, which replaces the unphysical part with a sharp saturation shock. We build $f_w$ from Corey relative permeabilities, find the shock with `Roots.jl`, and get its slope with `ForwardDiff.jl` — the same automatic-differentiation tool used for curve fitting elsewhere in this repository, now differentiating a closed-form flow function instead of an ODE solve.

Everything below is dimensionless (saturations, fractional flow, and pore volumes injected, $Q_i$) — the shock construction and breakthrough timing are unit-free by nature, so there's no field-unit conversion to get wrong.
"""

# ╔═╡ 18063c47-548d-4344-bc61-02955f2630a1
using ForwardDiff, Roots, PlutoUI, Plots

# ╔═╡ 6f16d8fd-9be5-4f52-b007-9aa8b7e9f09d
md"""
## 1. Relative permeability and fractional flow

Corey-type curves: water and oil relative permeability ramp from zero at their respective irreducible saturations.
"""

# ╔═╡ 4d83f7c7-69c5-4518-9635-4edceccaf6e6
function krw(Sw, Swc, Sor, krw_max, nw)
	Sw <= Swc && return 0.0
	Sw >= 1 - Sor && return krw_max
	Swn = (Sw - Swc) / (1 - Swc - Sor)
	return krw_max * Swn^nw
end

# ╔═╡ ad16f3af-d8cb-4c62-9843-37af1f77e5fa
function kro(Sw, Swc, Sor, kro_max, no)
	Sw <= Swc && return kro_max
	Sw >= 1 - Sor && return 0.0
	Son = (1 - Sw - Sor) / (1 - Swc - Sor)
	return kro_max * Son^no
end

# ╔═╡ bebfba8b-5790-43fc-aa63-d3e34483ab0d
function fw(Sw, p)
	Swc, Sor, krw_max, kro_max, nw, no, μw, μo = p
	kw = krw(Sw, Swc, Sor, krw_max, nw)
	ko = kro(Sw, Swc, Sor, kro_max, no)
	return (kw / μw) / (kw / μw + ko / μo)
end

# ╔═╡ e33a9ae7-d112-461b-a53d-c3ee0c6312d3
begin
	# Swc, Sor, krw_max, kro_max, nw, no, μw (cp), μo (cp)
	p_bl = (0.2, 0.2, 0.4, 0.9, 2.0, 2.0, 0.5, 2.0)
	Swc, Sor = p_bl[1], p_bl[2]
end

# ╔═╡ 806cf78c-e5a2-40b0-ab0c-3add0721e993
begin
	Sw_grid = range(Swc + 1e-4, 1 - Sor - 1e-4; length=200)
	plot(Sw_grid, fw.(Sw_grid, Ref(p_bl)); xlabel="Sw", ylabel="fw(Sw)", label="fractional flow",
		title="Fractional flow curve", lw=2)
end

# ╔═╡ c7d134c1-8437-4fe8-b1c5-258088e0b035
md"""
## 2. Welge's tangent construction: finding the shock front

The physically valid shock saturation $S_{wf}$ is where the straight line from $(S_{wc}, 0)$ is *tangent* to the $f_w$ curve — i.e. where the chord slope equals the local slope:

$$\frac{f_w(S_{wf})}{S_{wf} - S_{wc}} = f_w'(S_{wf})$$
"""

# ╔═╡ 5c0ac65b-a4f7-4b56-be03-725e966d4e6f
dfw(Sw, p) = ForwardDiff.derivative(s -> fw(s, p), Sw)

# ╔═╡ bcdf9f93-f0ad-4383-b2ce-f60467fedd04
begin
	tangent_gap(Sw) = fw(Sw, p_bl) / (Sw - Swc) - dfw(Sw, p_bl)
	Sw_front = find_zero(tangent_gap, (Swc + 1e-3, 1 - Sor - 1e-3))
	fw_front = fw(Sw_front, p_bl)
	dfw_front = dfw(Sw_front, p_bl)
	(Sw_front=Sw_front, fw_front=fw_front, dfw_front=dfw_front)
end

# ╔═╡ 3e899490-13af-4866-ac78-f18a1515889f
begin
	plot(Sw_grid, fw.(Sw_grid, Ref(p_bl)); label="fw(Sw)", xlabel="Sw", ylabel="fw", lw=2,
		title="Welge tangent construction")
	plot!([Swc, 1 - Sor], [0.0, dfw_front * (1 - Sor - Swc)]; label="tangent from (Swc, 0)", lw=2, ls=:dash)
	scatter!([Sw_front], [fw_front]; label="shock front Swf", ms=6, color=:red)
end

# ╔═╡ 98fb7ca0-62a4-46ee-8824-f2d81f10401e
md"""
## 3. Breakthrough and the average saturation behind the front

The front's characteristic velocity is proportional to $f_w'(S_{wf})$, so it reaches the outlet (dimensionless position $x_D = 1$) after $Q_{i,bt} = 1/f_w'(S_{wf})$ pore volumes injected. Welge's method also gives the average water saturation left behind the front with no extra integration.
"""

# ╔═╡ 56902aab-f7b1-4b93-a3fa-33ee0ba27f4a
begin
	Qi_breakthrough = 1 / dfw_front
	Sw_avg_behind = Sw_front + (1 - fw_front) / dfw_front
	(Qi_breakthrough=Qi_breakthrough, Sw_avg_behind_front=Sw_avg_behind)
end

# ╔═╡ e80b389a-8cf9-474e-871a-75eb31199259
md"""
## 4. The saturation profile ahead of breakthrough

Behind the shock (for saturations from $S_{wf}$ up to the injection-face value), each saturation travels at its own characteristic speed $x_D(S_w) = Q_i \, f_w'(S_w)$ — a valid, single-valued *rarefaction* wave. Ahead of the shock, only the original connate water $S_{wc}$ is present. Drag $Q_i$ to watch the flood front advance toward the outlet at $x_D=1$.
"""

# ╔═╡ d5eec91e-8ae4-4778-8b0d-f51cf229f5a2
@bind Qi_frac Slider(0.05:0.05:0.95, default=0.5, show_value=true)

# ╔═╡ 25d68072-5929-461d-b21f-49c7804fe072
begin
	Qi = Qi_frac * Qi_breakthrough   # stays pre-breakthrough: front hasn't reached the outlet yet

	Sw_behind = range(Sw_front, 1 - Sor; length=100)
	xD_behind = Qi .* dfw.(Sw_behind, Ref(p_bl))
	xD_front = Qi * dfw_front

	plot(xD_behind, Sw_behind; label="rarefaction wave", lw=2, xlabel="xD (dimensionless position)",
		ylabel="Sw", title="Saturation profile at Qi = $(round(Qi, digits=3)) PV", ylims=(0, 1))
	plot!([xD_front, xD_front], [Swc, Sw_front]; label="shock", lw=3, color=:red)
	plot!([xD_front, 1.0], [Swc, Swc]; label="undisturbed oil (Sw = Swc)", lw=2, color=:gray)
end

# ╔═╡ 429b016f-89de-4985-889c-1ba0d3964088
md"""
## Takeaways

* The Buckley–Leverett profile isn't solved with an ODE or PDE integrator at all here — `Roots.jl` finds the shock, and `ForwardDiff.jl` supplies the exact slope of a closed-form fractional-flow curve, the same automatic-differentiation building block used for curve fitting elsewhere in this repository.
* Working entirely in dimensionless saturations and pore-volumes-injected sidesteps unit-conversion risk altogether — the shock location and breakthrough timing are unit-free by construction.
* This is the textbook (Welge) construction for the simplest case — incompressible, 1D, no capillary pressure or gravity. A full reservoir simulator adds those effects numerically (finite-volume/finite-difference), but the fractional-flow curve built here is exactly the physics such a simulator has to get right first.
"""

# ╔═╡ Cell order:
# ╟─b7f36c98-470e-4738-9886-69364724fefe
# ╠═18063c47-548d-4344-bc61-02955f2630a1
# ╟─6f16d8fd-9be5-4f52-b007-9aa8b7e9f09d
# ╠═4d83f7c7-69c5-4518-9635-4edceccaf6e6
# ╠═ad16f3af-d8cb-4c62-9843-37af1f77e5fa
# ╠═bebfba8b-5790-43fc-aa63-d3e34483ab0d
# ╠═e33a9ae7-d112-461b-a53d-c3ee0c6312d3
# ╠═806cf78c-e5a2-40b0-ab0c-3add0721e993
# ╟─c7d134c1-8437-4fe8-b1c5-258088e0b035
# ╠═5c0ac65b-a4f7-4b56-be03-725e966d4e6f
# ╠═bcdf9f93-f0ad-4383-b2ce-f60467fedd04
# ╠═3e899490-13af-4866-ac78-f18a1515889f
# ╟─98fb7ca0-62a4-46ee-8824-f2d81f10401e
# ╠═56902aab-f7b1-4b93-a3fa-33ee0ba27f4a
# ╟─e80b389a-8cf9-474e-871a-75eb31199259
# ╠═d5eec91e-8ae4-4778-8b0d-f51cf229f5a2
# ╠═25d68072-5929-461d-b21f-49c7804fe072
# ╟─429b016f-89de-4985-889c-1ba0d3964088
