# Smaller pieces of API that the ported tatva tests do not reach.

using Test, Tatva, LinearAlgebra, SparseArrays, StaticArrays, ForwardDiff
using SparseConnectivityTracer: TracerLocalSparsityDetector

@testset "jacrev: argnums, scalar and array-valued functions" begin
	f(a, b) = sum(abs2, a) * b[1] + b[2]^3
	a, b = [1.0, 2.0], [3.0, 4.0]
	@test jacrev(f)(a, b) ≈ 2 .* a .* b[1]
	@test jacrev(f; argnums = 2)(a, b) ≈ [sum(abs2, a), 3 * b[2]^2]
	g(x) = [x[1] * x[2], sin(x[1])]
	@test jacrev(g)([1.0, 2.0]) ≈ ForwardDiff.jacobian(g, [1.0, 2.0])
	h(x::Real, y) = x^2 * y
	@test jacrev(h)(3.0, 2.0) ≈ 12.0
	@test_throws ArgumentError jacrev(f; backend = Tatva.AutoEnzyme())(a, b)  # Enzyme not loaded
end

@testset "local sparsity detector follows the branch taken" begin
	f(u) = u[1] > 0 ? sum(u[1:(end-1)] .* u[2:end]) : sum(abs2, u)
	n = 4
	@test_throws TypeError pattern_from_energy(f, n)  # the global tracer cannot branch
	tridiag = pattern_from_energy(f, n; detector = TracerLocalSparsityDetector(), x = ones(n))
	diag = pattern_from_energy(f, n; detector = TracerLocalSparsityDetector(), x = -ones(n))
	@test nnz(tridiag) == 2(n - 1)
	@test diag == sparse(I, n, n)
end

@testset "map_over_quads with array-valued element quantities" begin
	mesh = unit_square(2, 2)
	op = Operator(mesh, Tri3())
	nel = size(mesh.elements, 2)
	directions = reshape(collect(1.0:(2nel)), 2, nel)  # one 2-vector per element (element axis last)
	f(ξ, u, d) = dot(d, SVector(1.0, 1.0)) * sum(u)
	r = map_over_quads(op, f; element_quantity = (2,))(ones(num_nodes(mesh)), directions)
	@test vec(r) ≈ [3 * sum(directions[:, e]) for e in 1:nel]
	g(u, d) = d
	@test map_over_elements(op, g; element_quantity = (2,))(ones(num_nodes(mesh)), directions)[2] == SVector(3.0, 4.0)
end

@testset "integrate dispatch: constants, nodal and quadrature fields" begin
	mesh = unit_square(3, 3; type = :quad)
	op = Operator(mesh, Quad4())
	@test integrate(op, 2.0) ≈ 2.0
	@test integrate(op, fill(2.0, num_nodes(mesh))) ≈ 2.0
	@test integrate(op, fill(2.0, num_quad_points(Quad4()), size(mesh.elements, 2))) ≈ 2.0
	@test integrate(op, [SVector(1.0, 2.0) for _ in 1:num_nodes(mesh)]) ≈ [1.0, 2.0]
	@test sum(integrate_per_element(op, 1.0)) ≈ 1.0
	@test Tatva.remake(op; element = Quad4(Tatva.default_quadrature(Quad4)...)) isa Operator
end

@testset "lifter add, create_g2l, dof_range" begin
	l = Lifter(5, Fixed([1], RuntimeValue(:a, 1.0)))
	l = with_values(l, :a => 2.0)
	l2 = Tatva.add(l, Periodic([5], [4]))
	@test l2.size_reduced == 3
	@test lift_from_zeros(l2, [1.0, 2.0, 3.0]) == [2.0, 1.0, 2.0, 3.0, 3.0]  # value :a kept

	lookup = create_g2l([10, 4, 7])
	@test lookup([7, 10, 5, 4]) == [3, 1, 0, 2]

	@test Tatva.dof_range(10, 3, 0) == (0, 4)
	@test Tatva.dof_range(10, 3, 1) == (4, 7)
	@test Tatva.dof_range(10, 3, 2) == (7, 10)
end

@testset "compound: positional field arguments, shapes and errors" begin
	S = Compound(:S, :a => field(3, () -> [1.0, 2.0, 3.0]), :b => field((2, 2), nothing, SHARED))
	@test S().a == [1.0, 2.0, 3.0]
	@test S.b.field_type === SHARED
	@test sprint(show, S) == "S(a=(3,), b=(2, 2); size=7)"
	@test sprint(show, S()) == "S(a=(3,), b=(2, 2))"
	@test_throws CompoundError Tatva.get_layout(S)
	@test_throws CompoundError Tatva.global_indices(S)
end

@testset "ColoredMatrix from_csr alias and shape checks" begin
	A = sparse([1.0 2.0 0.0; 0.0 3.0 4.0])
	cm = Tatva.from_csr(A)
	@test to_dense(cm) == Matrix(A)
	@test_throws DimensionMismatch ColoredMatrix([1, 2], [1], [1.0], (2, 2))
end
