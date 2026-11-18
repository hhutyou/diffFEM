# End-to-end workflows (no tatva counterpart in its test suite; these mirror the examples of
# the tatva documentation): energy → gradient → sparse Hessian → Newton solve.

using Test, Tatva, LinearAlgebra, SparseArrays, StaticArrays, ForwardDiff

linear_elastic(∇u, μ, λ) = (ε = (∇u + ∇u') / 2; μ * dot(ε, ε) + λ / 2 * tr(ε)^2)

@testset "patch test: linear elasticity reproduces a linear field exactly" begin
	mesh = unit_square(6, 5)
	op = Operator(mesh, Tri3())
	Solution = Compound(:Solution, :u => field((2, AUTO)); mesh = mesh)
	A = [0.01 0.02; -0.015 0.03]
	exact = A * mesh.coords  # (2, n_nodes)
	x, y = mesh.coords[1, :], mesh.coords[2, :]
	boundary = findall((x .== 0) .| (x .== 1) .| (y .== 0) .| (y .== 1))
	dofs = Solution.u[:, boundary]
	lifter = Lifter(Solution.size, Fixed(dofs, vec(exact[:, boundary])))

	energy(u_full) = integrate(op, linear_elastic.(grad(op, Solution(u_full).u), 1.0, 2.0))
	energy_free(ur) = energy(lift_from_zeros(lifter, ur))
	cm = ColoredMatrix(adapt_sparsity(lifter, pattern_from_mesh(mesh, 2)))
	hessian = sparse_hessian(energy_free, cm)
	gradient_fn = jacrev(energy_free)

	ur = zeros(lifter.size_reduced)
	K = hessian(ur)
	ur -= spsolve(K, gradient_fn(ur))  # a single Newton step solves a quadratic energy
	u = Solution(lift_from_zeros(lifter, ur)).u
	@test u ≈ exact atol = 1e-12
	@test norm(gradient_fn(ur)) < 1e-12
end

@testset "hyperelasticity: Newton converges quadratically with the traced sparse Hessian" begin
	mesh = unit_square(4, 4; type = :quad)
	op = Operator(mesh, Quad4())
	Solution = Compound(:Solution, :u => field((2, AUTO)); mesh = mesh)
	μ, λ = 1.0, 2.0
	ψ(∇u) = (F = I + ∇u; J = det(F); μ / 2 * (tr(F' * F) - 2) - μ * log(J) + λ / 2 * log(J)^2)
	x = mesh.coords[1, :]
	left, right = findall(x .== 0), findall(x .== 1)
	lifter = Lifter(Solution.size, Fixed(Solution.u[:, left], 0.0), Fixed(Solution.u[1, right], RuntimeValue(:stretch, 0.0)))
	energy_free(ur, lf) = integrate(op, ψ.(grad(op, Solution(lift_from_zeros(lf, ur)).u)))

	cm = ColoredMatrix(pattern_from_energy(energy_free, lifter.size_reduced, lifter))
	@test cm.shape == (lifter.size_reduced, lifter.size_reduced)
	hessian = sparse_hessian(energy_free, cm)
	residual = jacrev(energy_free)

	ur = zeros(lifter.size_reduced)
	for stretch ∈ (0.1, 0.2)
		lf = with_values(lifter, :stretch => stretch)
		norms = Float64[]
		for _ ∈ 1:8
			r = residual(ur, lf)
			push!(norms, norm(r))
			norms[end] < 1e-11 && break
			ur -= spsolve(hessian(ur, lf), r)
		end
		@test norms[end] < 1e-11
		# quadratic convergence: the error squares from one iteration to the next
		k = findfirst(<(1e-3), norms)
		@test k !== nothing && k < length(norms) && norms[k+1] < 10 * norms[k]^2
	end
	# the right edge moved by the prescribed stretch, the reaction is a tensile force
	u = Solution(lift_from_zeros(with_values(lifter, :stretch => 0.2), ur)).u
	@test all(u[1, right] .≈ 0.2)
end

@testset "differentiable solve: compliance sensitivity to a material parameter" begin
	mesh = unit_square(4, 4)
	op = Operator(mesh, Tri3())
	n = 2 * num_nodes(mesh)
	x = mesh.coords[1, :]
	lifter = Lifter(n, Fixed(vcat(2 .* findall(x .== 0) .- 1, 2 .* findall(x .== 0)), 0.0))
	f = zeros(n)
	f[2 .* findall(x .== 1)] .= -0.01  # downward load on the right edge
	cm = ColoredMatrix(adapt_sparsity(lifter, pattern_from_mesh(mesh, 2)))
	function compliance(μ)
		E(ur) = integrate(op, linear_elastic.(grad(op, reshape(lift_from_zeros(lifter, ur), 2, :)), μ, 1.0)) -
				dot(reduce(lifter, f), ur)
		ur = zeros(eltype(μ), lifter.size_reduced)
		ur = ur - spsolve(sparse_hessian(E, cm)(ur), jacrev(E)(ur))
		return dot(reduce(lifter, f), ur)
	end
	c = compliance(1.0)
	@test c > 0
	dc = ForwardDiff.derivative(compliance, 1.0)
	h = 1e-6
	@test dc ≈ (compliance(1.0 + h) - compliance(1.0 - h)) / (2h) rtol = 1e-6
	@test dc < 0  # a stiffer material is less compliant
end
