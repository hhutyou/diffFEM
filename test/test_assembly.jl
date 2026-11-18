# Element-wise assembly (ElementAssembler) against the global AD path, and the lifter
# reduction of assembled operators.

using Test, Tatva, LinearAlgebra, SparseArrays, StaticArrays, ForwardDiff, Random

@testset "2D hyperelasticity: element-wise = global AD" begin
	mesh = unit_square(5, 4)
	op = Operator(mesh, Tri3())
	ψ(∇u) = (F = I + ∇u; J = det(F); 0.5 * (tr(F' * F) - 2) - log(J) + 0.6 * log(J)^2)
	density(ξ, Ue, Xe) = ψ(grad(op.element, ξ, Ue, Xe))
	E_global(u) = integrate(op, ψ.(grad(op, reshape(u, 2, :))))

	asm = ElementAssembler(op, 2)
	u = 0.02 .* randn(MersenneTwister(1), 2 * num_nodes(mesh))
	U = reshape(u, 2, :)
	E, g, K = assemble(asm, density, U, mesh.coords)
	@test E ≈ E_global(u)
	@test assemble_energy(asm, density, U, mesh.coords) ≈ E
	@test g ≈ ForwardDiff.gradient(E_global, u) atol = 1e-12
	@test assemble_gradient(asm, density, U, mesh.coords) ≈ g
	@test Matrix(K) ≈ ForwardDiff.hessian(E_global, u) atol = 1e-10
	@test assemble_hessian(asm, density, U, mesh.coords) ≈ K
	@test K.colptr == SparseMatrixCSC(pattern_from_mesh(mesh, 2)).colptr  # the mesh pattern
	@test_throws DimensionMismatch assemble(asm, density, vec(U), mesh.coords)  # 1 component, assembler 2
end

@testset "3D hyperelasticity on tetrahedra" begin
	coords = [0.0 1.0 0.0 0.0 1.0 1.0; 0.0 0.0 1.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0 1.0 1.0]
	mesh = Mesh(coords, [1 2 2; 2 3 5; 3 4 4; 4 5 6])
	op = Operator(mesh, Tetrahedron4())
	ψ(∇u) = (F = I + ∇u; J = det(F); 0.5 * (tr(F' * F) - 3) - log(J) + log(J)^2)
	density(ξ, Ue, Xe) = ψ(grad(op.element, ξ, Ue, Xe))
	E_global(u) = integrate(op, ψ.(grad(op, reshape(u, 3, :))))
	u = 0.01 .* randn(MersenneTwister(2), 3 * num_nodes(mesh))
	E, g, K = assemble(ElementAssembler(op, 3), density, reshape(u, 3, :), mesh.coords)
	@test E ≈ E_global(u)
	@test g ≈ ForwardDiff.gradient(E_global, u) atol = 1e-12
	@test Matrix(K) ≈ ForwardDiff.hessian(E_global, u) atol = 1e-10
end

@testset "scalar field with a nodal coefficient and an element quantity" begin
	mesh = unit_square(3, 3; type = :quad)
	op = Operator(mesh, Quad4())
	nel = size(mesh.elements, 2)
	κ = 1.0 .+ mesh.coords[1, :]       # nodal coefficient (not differentiated)
	s = collect(range(0.5, 2.0; length = nel))  # one source per element
	function density(ξ, ue, Xe, κe, se)
		el = op.element
		v = interpolate(el, ξ, ue, Xe)
		return 0.5 * interpolate(el, ξ, κe, Xe) * sum(abs2, grad(el, ξ, ue, Xe)) + 0.25 * v^4 - se * v
	end
	asm = ElementAssembler(op; element_quantity = (3,))
	function E_global(u)
		uq, κq, ∇uq = evaluate(op, u), evaluate(op, κ), grad(op, u)
		dens = [0.5 * κq[q, e] * sum(abs2, ∇uq[q, e]) + 0.25 * uq[q, e]^4 - s[e] * uq[q, e] for q in axes(uq, 1), e in axes(uq, 2)]
		return integrate(op, dens)
	end
	u = randn(MersenneTwister(3), num_nodes(mesh))
	E, g, K = assemble(asm, density, u, mesh.coords, κ, s)
	@test E ≈ E_global(u)
	@test assemble_energy(asm, density, u, mesh.coords, κ, s) ≈ E
	@test g ≈ ForwardDiff.gradient(E_global, u) atol = 1e-12
	@test Matrix(K) ≈ ForwardDiff.hessian(E_global, u) atol = 1e-10
	# extras of the wrong size are refused (the element gathers do not check bounds)
	@test_throws DimensionMismatch assemble(asm, density, u, mesh.coords, κ[1:(end-1)], s)
	@test_throws DimensionMismatch assemble_gradient(asm, density, u, mesh.coords, κ, s[1:(end-1)])
	@test_throws DimensionMismatch assemble_energy(asm, density, u, mesh.coords[:, 1:(end-1)], κ, s)
end

@testset "returned Hessians own their structure; in-place assembly" begin
	mesh = unit_square(4, 3)
	op = Operator(mesh, Tri3())
	ψ(∇u) = (F = I + ∇u; J = det(F); 0.5 * (tr(F' * F) - 2) - log(J) + 0.6 * log(J)^2)
	density(ξ, Ue, Xe) = ψ(grad(op.element, ξ, Ue, Xe))
	asm = ElementAssembler(op, 2)
	U = 0.02 .* randn(MersenneTwister(5), 2, num_nodes(mesh))
	E, g, K = assemble(asm, density, U, mesh.coords)
	K0 = copy(K)
	# structural changes to a returned matrix leave the assembler intact
	nonzeros(K)[1:2:end] .= 0
	dropzeros!(K)
	@test assemble_hessian(asm, density, U, mesh.coords) == K0
	@test nnz(asm.pattern) == nnz(K0)
	# assemble! / assemble_gradient! overwrite arrays of the assembler's structure
	g2, K2 = fill(NaN, length(g)), similar(asm.pattern)
	@test assemble!(g2, K2, asm, density, U, mesh.coords) ≈ E
	@test g2 ≈ g
	@test K2 ≈ K0
	@test assemble_gradient!(fill(NaN, length(g)), asm, density, U, mesh.coords) ≈ g
	# no allocation per element: a small constant per call (the field shape is a run-time value)
	big = unit_square(40, 40)
	asm_big = ElementAssembler(Operator(big, Tri3()), 2)
	U_big = zeros(2, num_nodes(big))
	_, g_big, K_big = assemble(asm_big, density, U_big, big.coords)
	assemble!(g2, K2, asm, density, U, mesh.coords)
	assemble!(g_big, K_big, asm_big, density, U_big, big.coords)
	@test @allocated(assemble!(g2, K2, asm, density, U, mesh.coords)) < 4096
	@test @allocated(assemble!(g_big, K_big, asm_big, density, U_big, big.coords)) < 4096
	@test_throws ArgumentError assemble!(g2, K, asm, density, U, mesh.coords)  # K lost entries
	@test_throws DimensionMismatch assemble!(zeros(3), K2, asm, density, U, mesh.coords)
	@test_throws DimensionMismatch assemble_gradient!(zeros(3), asm, density, U, mesh.coords)
end

@testset "element groups and threaded assembly ($(Threads.nthreads()) threads)" begin
	mesh = unit_square(40, 30)  # large enough for several tasks per group
	op = Operator(mesh, Tri3())
	ψ(∇u) = (F = I + ∇u; J = det(F); 0.5 * (tr(F' * F) - 2) - log(J) + 0.6 * log(J)^2)
	density(ξ, Ue, Xe) = ψ(grad(op.element, ξ, Ue, Xe))
	asm = ElementAssembler(op, 2)
	asm_t = ElementAssembler(op, 2; threaded = true)
	# every element in exactly one block, no two blocks of a group share a node
	@test sort!(reduce(vcat, collect.(reduce(vcat, asm.groups)))) == 1:num_elements(op)
	shares_no_node(group) = allunique(reduce(vcat, [unique(vec(mesh.elements[:, b])) for b in group]))
	@test all(shares_no_node, asm.groups)
	@test all(shares_no_node, Tatva._element_groups(mesh.elements; block = 1))  # element coloring
	x, y = mesh.coords[1, :], mesh.coords[2, :]
	U = 0.1 .* permutedims(hcat(sin.(3x) .* y, x .* y .^ 2))  # smooth: det F > 0
	E, g, K = assemble(asm, density, U, mesh.coords)
	Et, gt, Kt = assemble(asm_t, density, U, mesh.coords)
	@test Et ≈ E
	@test gt ≈ g
	@test Kt ≈ K
	@test assemble_energy(asm_t, density, U, mesh.coords) ≈ E
	@test assemble_gradient(asm_t, density, U, mesh.coords) ≈ g
end

@testset "lifting matrix and reduced operators" begin
	lifter = Lifter(10, Fixed([1, 5], 3.0), Periodic([2, 9], [4, 6]), Periodic([10], [2]))
	P = lifting_matrix(lifter)
	@test size(P) == (10, lifter.size_reduced)
	rng = MersenneTwister(4)
	for _ ∈ 1:3
		ur = randn(rng, lifter.size_reduced)
		@test P * ur + lift_from_zeros(lifter, zeros(lifter.size_reduced)) ≈ lift_from_zeros(lifter, ur)
		r = randn(rng, 10)
		@test transpose(P) * r ≈ reduce_adjoint(lifter, r)
	end
	# reduce_hessian is the Hessian of the lifted energy
	E(u) = sum(abs2, u) + u[2] * u[9]^2 + u[4]^2 * u[6]
	ur = randn(rng, lifter.size_reduced)
	H = ForwardDiff.hessian(E, lift_from_zeros(lifter, ur))
	@test Matrix(reduce_hessian(lifter, sparse(H))) ≈ ForwardDiff.hessian(v -> E(lift_from_zeros(lifter, v)), ur)
	@test reduce_hessian(lifter, H) ≈ reduce_hessian(lifter, sparse(H))  # dense input
	# Dirichlet conditions only: the submatrix of the free dofs
	ld = Lifter(10, Fixed([1, 5, 7], 3.0))
	@test reduce_hessian(ld, sparse(H)) == sparse(H)[ld.free_dofs, ld.free_dofs]
	@test_throws DimensionMismatch reduce_hessian(ld, sparse(H[1:9, 1:9]))
	# custom constraints fall back to a numerical construction
	struct Scale <: Tatva.Constraint
		dofs::Vector{Int}
		id::Tatva.UUID
	end
	Tatva.apply_lift!(c::Scale, u, rv) = (u[c.dofs] .= 2 .* u[c.dofs .- 1]; u)
	l2 = Lifter(4, Scale([3], Tatva.uuid4()))
	@test lifting_matrix(l2) * [1.0, 2.0, 3.0] ≈ lift_from_zeros(l2, [1.0, 2.0, 3.0])
	P2 = lifting_matrix(l2)
	@test reduce_hessian(l2, sparse(H[1:4, 1:4])) ≈ transpose(P2) * H[1:4, 1:4] * P2
end

@testset "Newton solve with assembled, reduced operators" begin
	mesh = unit_square(8, 8)
	op = Operator(mesh, Tri3())
	ψ(∇u) = (F = I + ∇u; J = det(F); 0.5 * (tr(F' * F) - 2) - log(J) + 0.6 * log(J)^2)
	density(ξ, Ue, Xe) = ψ(grad(op.element, ξ, Ue, Xe))
	Solution = Compound(:Solution, :u => field((2, AUTO)); mesh = mesh)
	x = mesh.coords[1, :]
	left, right = findall(x .== 0), findall(x .== 1)
	lifter = Lifter(Solution.size, Fixed(Solution.u[:, left], 0.0), Fixed(Solution.u[1, right], 0.1))
	asm = ElementAssembler(op, 2)
	ur = zeros(lifter.size_reduced)
	norms = Float64[]
	for _ ∈ 1:10
		u = lift_from_zeros(lifter, ur)
		_, g, K = assemble(asm, density, reshape(u, 2, :), mesh.coords)
		r = reduce_adjoint(lifter, g)
		push!(norms, norm(r))
		norms[end] < 1e-10 && break
		ur -= spsolve(reduce_hessian(lifter, K), r)
	end
	@test norms[end] < 1e-10
	# the same residual as the global AD path
	E_free(ur) = integrate(op, ψ.(grad(op, reshape(lift_from_zeros(lifter, ur), 2, :))))
	@test norm(jacrev(E_free)(ur)) < 1e-9
end
