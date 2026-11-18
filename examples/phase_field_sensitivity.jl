# Phase-field fracture (AT2, history field, staggered scheme) of a single-edge notched plate in
# tension (after Miehe et al. 2010), and the derivative of the whole force–displacement curve
# with respect to the fracture toughness G_c: the same simulation run with a ForwardDiff dual
# number differentiates through the irreversible crack growth, sparse solves included.
#
#     julia --project=. -t auto examples/phase_field_sensitivity.jl
#
# Up to the peak load the derivative agrees with central differences; across the unstable
# crack jump the force is not a smooth function of G_c (and the derivative of the staggered
# fixed point lags its value), so derivatives there are not meaningful.

using Tatva, LinearAlgebra, SparseArrays, ForwardDiff, Printf

const λ, μ = 121.15, 80.77  # plane strain, kN/mm² (E = 210, ν = 0.3)
const ℓ = 0.035             # length scale, mm (≈ 2 elements)
const k_res = 1e-8          # residual stiffness
ψ(∇u) = (ε = (∇u + ∇u') / 2; λ / 2 * tr(ε)^2 + μ * dot(ε, ε))

# 1 mm × 1 mm plate with a slit from the left edge to the centre: the nodes on the slit are
# doubled, the elements above it use the copies
function slit_square(n)
	m = unit_square(n, n)
	X, E = copy(m.coords), copy(m.elements)
	on_slit = findall((abs.(X[2, :] .- 0.5) .< 1e-12) .& (X[1, :] .< 0.5 - 1e-12))
	copies = Dict(k => size(X, 2) + i for (i, k) ∈ enumerate(on_slit))
	X = hcat(X, X[:, on_slit])
	for e ∈ axes(E, 2)
		sum(X[2, E[:, e]]) / 3 > 0.5 && (E[:, e] = [get(copies, k, k) for k in E[:, e]])
	end
	return Mesh(X, E)
end

const mesh = slit_square(60)
const op = Operator(mesh, Tri3())  # one quadrature point: one history value per element
dof(c, nodes) = 2 .* (nodes .- 1) .+ c
const bottom, top = findall(mesh.coords[2, :] .== 0), findall(mesh.coords[2, :] .== 1)

# clamped bottom, top pulled up by δ (and held horizontally); the damage is free
lifter_u(δ) = Lifter(2 * num_nodes(mesh), Fixed(vcat(dof(1, bottom), dof(2, bottom), dof(1, top)), 0.0), Fixed(dof(2, top), δ))
const lifter_d = Lifter(num_nodes(mesh))

const asm_u = ElementAssembler(op, 2; threaded = true)
const asm_d = ElementAssembler(op, 1; element_quantity = (2,), threaded = true)  # extras: coords, history
density_u(ξ, Ue, Xe, de) = ((1 - interpolate(op.element, ξ, de, Xe))^2 + k_res) * ψ(grad(op.element, ξ, Ue, Xe))

# minimizer of an energy quadratic in the free dofs: one Newton step from the lift of zero
function solve_quadratic(asm, density, lifter, extras...)
	T = promote_type(map(eltype, extras)...)
	u0 = lift_from_zeros(lifter, zeros(T, lifter.size_reduced))
	field = asm === asm_u ? reshape(u0, 2, :) : u0
	_, g, K = assemble(asm, density, field, extras...)
	return lift_from_zeros(lifter, -spsolve(reduce_hessian(lifter, K), reduce_adjoint(lifter, g)))
end

"""Force on the top edge at every load step for the fracture toughness `Gc` (a number or a dual number)."""
function simulate(Gc, δs; tol = 1e-6, max_staggered = 500)
	# (local names differ from the variables of `simulate`: a closure assigning `d` would rebind them)
	density_d(ξ, de, Xe, He) = (dq = interpolate(op.element, ξ, de, Xe); ∇dq = grad(op.element, ξ, de, Xe);
		(1 - dq)^2 * He + Gc / (2ℓ) * (dq^2 + ℓ^2 * dot(∇dq, ∇dq)))
	T = typeof(Gc)
	d = zeros(T, num_nodes(mesh))
	H = zeros(T, num_elements(mesh))  # history: largest elastic energy density so far
	U = zeros(T, 2, num_nodes(mesh))
	forces = T[]
	for δ ∈ δs
		for _ ∈ 1:max_staggered
			U = reshape(solve_quadratic(asm_u, density_u, lifter_u(δ), mesh.coords, d), 2, :)
			H = max.(H, ψ.(vec(grad(op, U))))  # irreversibility
			d_new = solve_quadratic(asm_d, density_d, lifter_d, mesh.coords, H)
			change = maximum(abs, ForwardDiff.value.(d_new .- d))
			d = d_new
			change < tol && break
		end
		g = assemble_gradient(asm_u, density_u, U, mesh.coords, d)
		push!(forces, sum(g[dof(2, top)]))  # reaction on the top edge
	end
	return forces
end

const Gc = 2.7e-3  # kN/mm
δs = 0.00025:0.00025:0.0075
simulate(Gc, δs[1:2])  # compile
t = @elapsed F = simulate(Gc, δs)
td = @elapsed dF = ForwardDiff.derivative(G -> simulate(G, δs), Gc)  # the whole curve's sensitivity
h = 1e-4 * Gc
fd = (simulate(Gc + h, δs) .- simulate(Gc - h, δs)) ./ 2h

k = argmax(F)
@printf("%d elements, %d load steps: one run %.1f s, with the derivative %.1f s\n", num_elements(mesh), length(δs), t, td)
@printf("peak force %.4f kN at δ = %.5f mm; after the crack jump F = %.4f kN\n", F[k], δs[k], F[end])
println("  δ [mm]    F [kN]   dF/dGc: dual numbers   central differences")
for i ∈ unique(vcat(4:4:k, k))
	@printf("  %.5f  %8.4f  %18.4f  %18.4f\n", δs[i], F[i], dF[i], fd[i])
end
