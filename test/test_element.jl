# Port of tatva/tests/test_element.py, plus Julia-specific checks of the element library.

using Test, Tatva, StaticArrays, LinearAlgebra, Random, ForwardDiff

# Physical coordinates: lines are embedded along the x-axis of 2D space (one column per
# node); other elements use their reference nodes, so affine elements have J = I and any
# linear field is represented exactly.
const LINE2_COORDS = [0.0 1.0; 0.0 0.0]           # x ∈ [0, 1]
const LINE3_COORDS = [0.0 1.0 0.5; 0.0 0.0 0.0]   # ends then midpoint
ref_coords(E) = reference_nodes(E())

const ALL_ELEMENTS = (Line2, Line3, Tri3, Tri6, Quad4, Quad8, Tetrahedron4, Hexahedron8)
const SOLID_ELEMENTS = (Tri3, Tri6, Quad4, Quad8, Tetrahedron4, Hexahedron8)

@testset "scalar gradient of a linear field is exact: $E" for (E, coords, coeffs) in (
	(Line2, LINE2_COORDS, [3.0]),
	(Line3, LINE3_COORDS, [3.0]),
	(Tri3, ref_coords(Tri3), [2.0, 3.0]),
	(Tri6, ref_coords(Tri6), [2.0, 3.0]),
	(Quad4, ref_coords(Quad4), [2.0, 3.0]),
	(Quad8, ref_coords(Quad8), [2.0, 3.0]),
	(Tetrahedron4, ref_coords(Tetrahedron4), [2.0, 3.0, 4.0]),
	(Hexahedron8, ref_coords(Hexahedron8), [2.0, 3.0, 4.0]),
)
	el = E()
	if E in (Line2, Line3)
		u = 3.0 .* coords[1, :]  # u = 3x: the arc-length derivative is 3
		expected = 3.0
	else
		u = coords' * coeffs     # u = a · x  ⇒  ∇u = a
		expected = coeffs
	end
	for ξ ∈ el.quad_points
		@test gradient(el, ξ, u, coords) ≈ expected atol = 1e-12
	end
end

@testset "scalar hessian is exact (Quad8)" begin
	el = Quad8()
	coords = reference_nodes(el)
	x, y = coords[1, :], coords[2, :]
	u = x .^ 2 .+ 2.0 .* x .* y .+ 3.0 .* y .^ 2
	for ξ ∈ el.quad_points
		@test hessian(el, ξ, u, coords) ≈ [2.0 2.0; 2.0 6.0] atol = 1e-12
	end
end

@testset "vector gradient is exact (component first): $E" for E in SOLID_ELEMENTS
	el = E()
	coords = reference_nodes(el)
	dim = size(coords, 1)
	A = randn(MersenneTwister(0), dim, dim)
	u = A * coords  # u[:, n] = A x_n  ⇒  ∂uᵢ/∂xⱼ = A[i, j]
	for ξ ∈ el.quad_points
		@test gradient(el, ξ, u, coords) ≈ A atol = 1e-12
	end
end

@testset "tensor gradient is exact (value axes first, spatial last): $E" for E in SOLID_ELEMENTS
	el = E()
	coords = reference_nodes(el)
	dim, n = size(coords)
	B = randn(MersenneTwister(42), 2, 2, dim)
	T = reshape(reshape(B, 4, dim) * coords, 2, 2, n)  # T[i, j, n] = Σₖ B[i, j, k] x[k, n]
	for ξ ∈ el.quad_points
		@test gradient(el, ξ, T, coords) ≈ B atol = 1e-12
	end
end

@testset "vector hessian on a curved element matches physical finite differences" begin
	el = Quad8()
	# perturb the midside nodes: the mapping is no longer affine
	X = Matrix(reference_nodes(el))
	X[:, 5:8] .+= 0.05 .* [0.3 -0.2 0.1 0.4; 0.2 0.1 -0.3 0.25]
	x, y = X[1, :], X[2, :]
	u = permutedims(hcat(x .^ 2 .- x .* y, 3 .* y .^ 2 .+ x))  # nodal values (2, 8)
	ξ0 = SVector(0.2, -0.3)
	H = hessian(el, ξ0, u, X)
	@test size(H) == (2, 2, 2)
	@test H[:, 1, 2] ≈ H[:, 2, 1] atol = 1e-10  # symmetric in the spatial axes

	# independent check: ∂(∇u)/∂xⱼ by central differences, mapping x → ξ with Newton
	x0 = X * shape_function(el, ξ0)
	function ξ_of(xp)
		ξ = ξ0
		for _ ∈ 1:30
			r = X * shape_function(el, ξ) - xp
			ξ -= (X * transpose(shape_function_derivative(el, ξ))) \ r
		end
		return ξ
	end
	h = 1e-5
	for j ∈ 1:2
		e = SVector(ntuple(k -> k == j ? h : 0.0, 2))
		dg = (gradient(el, ξ_of(x0 + e), u, X) - gradient(el, ξ_of(x0 - e), u, X)) / (2h)
		@test H[:, :, j] ≈ dg atol = 1e-6
	end
end

@testset "shape functions: partition of unity, Kronecker property: $E" for E in ALL_ELEMENTS
	el = E()
	R = reference_nodes(el)
	N = nodes_per_element(el)
	for ξ ∈ el.quad_points
		@test sum(shape_function(el, ξ)) ≈ 1.0 atol = 1e-14
	end
	for j ∈ 1:N
		@test shape_function(el, R[:, j]) ≈ [i == j ? 1.0 : 0.0 for i in 1:N] atol = 1e-14
	end
end

@testset "shape function derivatives match AD: $E" for E in ALL_ELEMENTS
	el = E()
	D = reference_dim(el)
	for ξ ∈ (el.quad_points..., SVector{D}(ntuple(i -> 0.1 * i, D)))
		dN_ad = ForwardDiff.jacobian(η -> shape_function(el, η), ξ)  # (N, D)
		@test shape_function_derivative(el, ξ) ≈ transpose(dN_ad) atol = 1e-13
	end
end

@testset "default quadrature integrates the reference measure: $E" for (E, measure) in (
	(Line2, 2.0), (Line3, 2.0), (Tri3, 0.5), (Tri6, 0.5), (Quad4, 4.0), (Quad8, 4.0), (Tetrahedron4, 1 / 6), (Hexahedron8, 8.0),
)
	@test sum(E().quad_weights) ≈ measure
end

@testset "Tri6 quadrature integrates quadratics exactly" begin
	el = Tri6()
	# ∫_T x² = 1/12, ∫_T xy = 1/24 on the reference triangle
	@test sum(w * ξ[1]^2 for (ξ, w) in zip(el.quad_points, el.quad_weights)) ≈ 1 / 12
	@test sum(w * ξ[1] * ξ[2] for (ξ, w) in zip(el.quad_points, el.quad_weights)) ≈ 1 / 24
end

@testset "custom quadrature and value-based equality" begin
	@test Tri3() == Tri3()
	@test hash(Tri3()) == hash(Tri3())
	@test Tri3() != Quad4()
	pts = [1/6 2/3 1/6; 1/6 1/6 2/3]  # (D, n_q)
	el = Tri3(pts, [1 / 6, 1 / 6, 1 / 6])
	@test num_quad_points(el) == 3
	@test el != Tri3()
	@test el == Tri3([SVector(1 / 6, 1 / 6), SVector(2 / 3, 1 / 6), SVector(1 / 6, 2 / 3)], fill(1 / 6, 3))
	@test_throws DimensionMismatch Tri3(pts, [1.0])
end

@testset "get_jacobian and get_local_values" begin
	el = Quad4()
	X = [0.0 2.0 2.0 0.0; 0.0 0.0 1.0 1.0]  # 2×1 rectangle
	J, detJ = get_jacobian(el, SVector(0.1, 0.2), X)
	@test J ≈ [1.0 0.0; 0.0 0.5]
	@test detJ ≈ 0.5
	u = X[1, :] .+ X[2, :]
	val, g, d = get_local_values(el, SVector(0.0, 0.0), u, X)
	@test val ≈ 1.5
	@test g ≈ [1.0, 1.0]
	@test d ≈ 0.5
	# lines report the length element along the curve
	J, detJ = get_jacobian(Line2(), SVector(0.0), [0.0 3.0; 0.0 4.0])
	@test J ≈ 2.5 && detJ ≈ 2.5
end

@testset "triangle embedded in 3D: area element and tangential gradient" begin
	el = Tri3()
	X = [0.0 1.0 0.0; 0.0 0.0 1.0; 0.0 0.0 1.0]  # tilted: area √2 / 2
	_, detJ = get_jacobian(el, el.quad_points[1], X)
	@test detJ * sum(el.quad_weights) ≈ sqrt(2) / 2
	u = X' * [1.0, 2.0, 2.0]  # u = x + 2y + 2z, gradient tangential to the plane y = z
	g = gradient(el, el.quad_points[1], u, X)
	@test g ≈ [1.0, 2.0, 2.0] atol = 1e-12
end

@testset "element kernels are allocation free and type stable" begin
	for el ∈ (Tri3(), Quad4(), Tetrahedron4(), Hexahedron8(), Quad8())
		X = reference_nodes(el) .+ 0.01
		V = SMatrix{size(X)...}(X)  # a vector field
		ξ = el.quad_points[1]
		@inferred gradient(el, ξ, V, X)
		@inferred get_jacobian(el, ξ, X)
		@inferred interpolate(el, ξ, V, X)
		gradient(el, ξ, V, X)
		@test (@allocated gradient(el, ξ, V, X)) == 0
		@test (@allocated get_jacobian(el, ξ, X)) == 0
	end
end
