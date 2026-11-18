# Port of tatva/tests/test_operator.py, plus tests of the Julia-specific behaviour (dual
# numbers in the geometry, cached weights, interpolation on curved and 3D elements).

using Test, Tatva, StaticArrays, LinearAlgebra, ForwardDiff, Random

const NODES = [0.0 1.0 1.0 0.0; 0.0 0.0 1.0 1.0]  # (2, 4)
const ELEMENTS = [1 1; 2 3; 3 4]                  # (3, 2): triangles [1,2,3] and [1,3,4]
const EXPECTED_ELEMENT_AREAS = [0.5, 0.5]

tri_operator() = Operator(Mesh(NODES, ELEMENTS), Tri3())

function manual_map_results(f, nodal_values)
	el = Tri3()
	return [f(ξ, nodal_values[ELEMENTS[:, e]]) for ξ in el.quad_points, e in 1:size(ELEMENTS, 2)]
end

@testset "map applies the function per element and quadrature point" begin
	op = tri_operator()
	u = collect(0.0:3.0)
	f(ξ, el_values) = sum(el_values) + sum(ξ)
	result = map_over_quads(op, f)(u)
	@test size(result) == (1, 2)
	@test result ≈ manual_map_results(f, u)
end

@testset "map respects element_quantity" begin
	op = tri_operator()
	u = collect(0.0:3.0)
	bias = [10.0, 20.0]
	f(ξ, el_values, b) = sum(el_values) + b + ξ[1]
	result = map_over_quads(op, f; element_quantity = (2,))(u, bias)
	expected = [sum(u[ELEMENTS[:, e]]) + bias[e] + ξ[1] for ξ in Tri3().quad_points, e in 1:2]
	@test result ≈ expected
end

@testset "map passes keyword arguments and nodal fields as static arrays" begin
	op = tri_operator()
	U = copy(NODES)  # a vector field: element values are SMatrix{2,3}
	f(ξ, el_coords; scale) = scale * sum(el_coords)
	r = map_over_quads(op, f)(U; scale = 2.0)
	@test r ≈ [2 * sum(NODES[:, ELEMENTS[:, e]]) for _ in 1:1, e in 1:2]
	g(ξ, V) = V
	@test map_over_quads(op, g)(U)[1, 2] isa SMatrix{2, 3}
end

@testset "map_over_elements matches manual" begin
	op = tri_operator()
	u = collect(0.0:3.0)
	result = map_over_elements(op, sum)(u)
	@test result ≈ [sum(u[ELEMENTS[:, e]]) for e in 1:2]
	# element quantities are passed through
	r2 = map_over_elements(op, (v, k) -> sum(v) * k; element_quantity = (2,))(u, [1.0, 10.0])
	@test r2 ≈ [sum(u[ELEMENTS[:, 1]]), 10 * sum(u[ELEMENTS[:, 2]])]
end

@testset "evaluate returns the value at the centroid" begin
	op = tri_operator()
	u = [0.0, 1.0, 2.0, 3.0]
	@test evaluate(op, u) ≈ [1.0 5/3]
end

@testset "gradient of a linear field is constant" begin
	op = tri_operator()
	u = NODES' * [2.0, 3.0]
	g = grad(op, u)
	@test size(g) == (1, 2)
	@test all(gi -> gi ≈ [2.0, 3.0], g)
end

@testset "hess maps a nonzero hessian over the quadrature points" begin
	el = Quad8()
	coords = Matrix(reference_nodes(el))
	op = Operator(Mesh(coords, reshape(collect(1:8), 8, 1)), el)
	x, y = coords[1, :], coords[2, :]
	u = x .^ 2 .+ 2.0 .* x .* y .+ 3.0 .* y .^ 2
	H = hess(op, u)
	@test size(H) == (num_quad_points(el), 1)
	@test all(h -> isapprox(h, [2.0 2.0; 2.0 6.0]; atol = 1e-12), H)
end

@testset "integrate a nodal field matches the area" begin
	op = tri_operator()
	u = ones(4)
	@test integrate_per_element(op, u) ≈ EXPECTED_ELEMENT_AREAS
	@test integrate(op, u) ≈ sum(EXPECTED_ELEMENT_AREAS)
	@test integrate(op, 1.0) ≈ 1.0
end

@testset "integrate quadrature values matches manual" begin
	op = tri_operator()
	q = fill(4.0, 1, 2)
	@test integrate_per_element(op, q) ≈ 4 .* EXPECTED_ELEMENT_AREAS
	# array-valued integrands give array-valued integrals
	qv = fill(SVector(1.0, 2.0), 1, 2)
	@test integrate(op, qv) ≈ [1.0, 2.0]
end

@testset "interpolate recovers a linear function" begin
	op = tri_operator()
	u = vec(sum(NODES; dims = 1))
	points = [0.25 0.75 0.25 0.5; 0.25 0.25 0.75 0.5]
	@test interpolate(op, u, points) ≈ vec(sum(points; dims = 1))
	# vector fields interpolate to static vectors
	@test interpolate(op, NODES, points) ≈ [SVector(points[:, k]...) for k in 1:4]
end

@testset "interpolate raises for points outside the mesh" begin
	op = tri_operator()
	@test_throws ErrorException interpolate(op, ones(4), reshape([1.5, 0.5], 2, 1))
end

@testset "make_interpolate: bilinear quads and tetrahedra (Newton inverse map)" begin
	# a distorted quad: one Newton step from the centroid is not exact here
	mesh = Mesh([0.0 2.0 2.5 -0.2; 0.0 0.1 1.8 1.0], reshape([1, 2, 3, 4], 4, 1))
	op = Operator(mesh, Quad4())
	u = mesh.coords[1, :] .* mesh.coords[2, :]  # bilinear in ξ? no: x*y is not in Q1 on a distorted quad
	ξ = SVector(0.3, -0.4)
	p = mesh.coords * shape_function(Quad4(), ξ)
	f = make_interpolate(op, reshape(p, 2, 1))
	@test f(mesh.coords[1, :])[1] ≈ p[1]  # linear fields are reproduced exactly
	@test f(u)[1] ≈ dot(shape_function(Quad4(), ξ), u)

	tet = Mesh([0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0], reshape([1, 2, 3, 4], 4, 1))
	opt = Operator(tet, Tetrahedron4())
	pts = [0.1 0.2; 0.2 0.1; 0.3 0.25]
	@test interpolate(opt, vec(sum(tet.coords; dims = 1)), pts) ≈ vec(sum(pts; dims = 1))
end

random_reference_point(::Tri6, rng) = (ξ = rand(rng, SVector{2}); sum(ξ) > 1 ? 1 .- ξ : ξ)
random_reference_point(::Quad8, rng) = 2 .* rand(rng, SVector{2}) .- 1

@testset "make_interpolate: curved elements reaching beyond their nodes" begin
	# a Tri6 whose bottom edge bulges below all its nodes
	el = Tri6()
	X = Matrix(reference_nodes(el))
	X[2, 4], X[2, 2], X[2, 5] = 0.05, 0.6, 0.8
	op = Operator(Mesh(X, reshape(collect(1:6), 6, 1)), el)
	ξ = SVector(0.2, 0.001)
	p = X * shape_function(el, ξ)
	@test p[2] < minimum(X[2, :])
	u = collect(1.0:6.0) .^ 2
	@test make_interpolate(op, reshape(p, 2, 1))(u)[1] ≈ dot(shape_function(el, ξ), u)
	# the control points bound the whole element (random curved Tri6 and Quad8)
	rng = MersenneTwister(5)
	for el ∈ (Tri6(), Quad8()), _ ∈ 1:20
		Xe = reference_nodes(el) + 0.3 * (rand(rng, typeof(reference_nodes(el))) .- 0.5)
		lo, hi = Tatva._element_box(el, Xe)
		@test all(all(lo .<= Xe * shape_function(el, random_reference_point(el, rng)) .<= hi) for _ in 1:200)
	end
end

@testset "make_interpolate: grid search, lowest element index on shared boundaries" begin
	# a graded mesh: elements of very different sizes share the search grid
	mesh = unit_square(9, 7)
	mesh = Mesh(mesh.coords .^ 3, mesh.elements)
	op = Operator(mesh, Tri3())
	located = Tatva._locate_points(op, [SVector(mesh.coords[1, k], mesh.coords[2, k]) for k in 1:num_nodes(mesh)], 1e-8)
	@test all(first(located[k]) == findfirst(e -> k in mesh.elements[:, e], 1:num_elements(mesh)) for k in 1:num_nodes(mesh))
	pts = rand(MersenneTwister(6), 2, 500)
	u = 2 .* mesh.coords[1, :] .- mesh.coords[2, :]
	@test interpolate(op, u, pts) ≈ 2 .* pts[1, :] .- pts[2, :]
	@test_throws ErrorException interpolate(op, u, [0.5 1.2; 0.5 0.5])
end

@testset "map validates the sizes of its arguments" begin
	op = tri_operator()
	f = map_over_quads(op, (ξ, u, s) -> u[1] * s; element_quantity = (2,))
	@test size(f(ones(4), ones(2))) == (num_quad_points(op), 2)
	@test_throws DimensionMismatch f(ones(3), ones(2))
	@test_throws DimensionMismatch f(ones(4), ones(3))
	@test_throws DimensionMismatch map_over_elements(op, U -> sum(U))(ones(2, 5))
	@test_throws DimensionMismatch map_over_quads(op, (ξ, v) -> v[1])([SVector(1.0, 2.0)])
end

@testset "quads are the physical quadrature points" begin
	op = tri_operator()
	q = quads(op)
	@test q[1, 1] ≈ [2 / 3, 1 / 3]
	@test q[1, 2] ≈ [1 / 3, 2 / 3]
end

@testset "cached weights give identical results" begin
	mesh = unit_square(3, 3; type = :quad)
	op = Operator(mesh, Quad4())
	opc = Operator(mesh, Quad4(); cache_weights = true)
	@test get_integration_weights(op) ≈ get_integration_weights(opc)
	u = mesh.coords[1, :] .^ 2
	@test integrate(op, u) ≈ integrate(opc, u)
	# the Q1 interpolant of x² on cells of width 1/3 integrates to Σ h (a² + b²) / 2 = 19/54
	@test integrate(op, u) ≈ 19 / 54
end

@testset "operator validates the mesh" begin
	@test_throws ArgumentError Operator(Mesh(NODES, [1 1; 2 3; 3 5]), Tri3())  # node 5 does not exist
	@test_throws ArgumentError Operator(Mesh(NODES, [0 1; 2 3; 3 4]), Tri3())  # 1-based
	@test_throws ArgumentError Operator(Mesh(NODES, ELEMENTS), Quad4())        # wrong nodes per element
	@test_throws ArgumentError Operator(Mesh(NODES, zeros(Int, 3, 0)), Tri3())
	@test_throws DimensionMismatch grad(tri_operator(), ones(5))
end

# tatva: "operator can be used as a dynamic jit argument" (a pytree whose coordinates are
# traced). The Julia analogue: the geometry can carry dual numbers, so shape derivatives work.
@testset "operator with dual-valued coordinates (shape derivatives)" begin
	mesh = unit_square(3, 3)
	area(X) = integrate(Operator(Mesh(reshape(X, 2, :), mesh.elements), Tri3()), 1.0)
	X0 = vec(copy(mesh.coords))
	@test area(X0) ≈ 1.0
	dA = ForwardDiff.gradient(area, X0)
	# moving the right boundary outwards increases the area by its length
	right = findall(mesh.coords[1, :] .== 1.0)
	@test sum(dA[2 .* right .- 1]) ≈ 1.0
	# interior nodes do not change the area
	interior = findall(n -> 0 < mesh.coords[1, n] < 1 && 0 < mesh.coords[2, n] < 1, 1:num_nodes(mesh))
	@test all(abs.(dA[2 .* interior .- 1]) .< 1e-12)
end

# The component count of a dense `(n_comp, n_nodes)` field is runtime information, so a call
# with a plain matrix dispatches once through a function barrier; scalar fields and vectors
# of static arrays carry their value shape in the type and are inferred end to end.
@testset "operator entry points are type stable" begin
	mesh = unit_square(4, 4; type = :quad)
	op = Operator(mesh, Quad4())
	n = num_nodes(mesh)
	u = rand(n)
	Us = [SVector(rand(), rand()) for _ in 1:n]
	@test (@inferred grad(op, u)) isa Matrix{SVector{2, Float64}}
	@test (@inferred grad(op, Us)) isa Matrix{SMatrix{2, 2, Float64, 4}}
	@test (@inferred evaluate(op, Us)) isa Matrix{SVector{2, Float64}}
	@test (@inferred integrate(op, u)) isa Float64
	@inferred get_integration_weights(op)
	@test grad(op, Us) == grad(op, reduce(hcat, Us))
	# the element loop allocates only the output
	grad(op, u)
	@test (@allocated grad(op, u)) <= 2 * sizeof(grad(op, u)) + 512
end
