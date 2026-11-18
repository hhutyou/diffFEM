# Port of tatva/tests/test_operator_projection.py

using Test, Tatva, StaticArrays, SparseArrays, LinearAlgebra, ForwardDiff

proj_mesh() = unit_square(2, 2; type = :quad)
proj_op(mesh) = Operator(mesh, Quad4())

@testset "project a scalar field" begin
	mesh = proj_mesh()
	op = proj_op(mesh)
	field = [x[1] + x[2] for x in quads(op)]  # s(x, y) = x + y
	cm = ColoredMatrix(pattern_from_mesh(mesh, 1))
	projected = project(op, field, cm)
	@test size(projected) == (num_nodes(mesh),)
	@test projected ≈ mesh.coords[1, :] .+ mesh.coords[2, :] atol = 1e-7
end

@testset "project a vector field" begin
	mesh = proj_mesh()
	op = proj_op(mesh)
	field = quads(op)  # v(x, y) = (x, y), a Matrix{SVector{2}}

	# scalar matrix: one right-hand side per component
	projected_scalar = project(op, field, ColoredMatrix(pattern_from_mesh(mesh, 1)))
	@test size(projected_scalar) == (2, num_nodes(mesh))
	@test projected_scalar ≈ mesh.coords atol = 1e-7

	# vector matrix: coupled solve
	projected_vector = project(op, field, ColoredMatrix(pattern_from_mesh(mesh, 2)))
	@test size(projected_vector) == (2, num_nodes(mesh))
	@test projected_vector ≈ mesh.coords atol = 1e-7
end

@testset "project a tensor field" begin
	mesh = proj_mesh()
	op = proj_op(mesh)
	field = [SMatrix{2, 2}(x[1], 0.0, 0.0, x[2]) for x in quads(op)]  # T = [x 0; 0 y]
	projected = project(op, field, ColoredMatrix(pattern_from_mesh(mesh, 1)))
	@test size(projected) == (2, 2, num_nodes(mesh))
	@test projected[1, 1, :] ≈ mesh.coords[1, :] atol = 1e-7
	@test projected[2, 2, :] ≈ mesh.coords[2, :] atol = 1e-7
	@test all(abs.(projected[1, 2, :]) .< 1e-7)
	@test all(abs.(projected[2, 1, :]) .< 1e-7)
end

@testset "project with the default matrix" begin
	mesh = proj_mesh()
	op = proj_op(mesh)
	field = [x[1] for x in quads(op)]
	projected = project(op, field)
	@test size(projected) == (num_nodes(mesh),)
	@test projected ≈ mesh.coords[1, :] atol = 1e-7
end

@testset "project with a lifter (fixed values are set, the rest is projected)" begin
	mesh = unit_square(3, 3; type = :quad)
	op = proj_op(mesh)
	field = [x[1] + 2x[2] for x in quads(op)]
	left = findall(mesh.coords[1, :] .== 0.0)
	exact = mesh.coords[1, :] .+ 2 .* mesh.coords[2, :]
	lifter = Lifter(num_nodes(mesh), Fixed(left, exact[left]))
	projected = project(op, field; lifter = lifter)
	@test projected ≈ exact atol = 1e-7
	# the same with an explicit (reduced) colored matrix
	cm = ColoredMatrix(adapt_sparsity(lifter, pattern_from_mesh(mesh, 1)))
	@test project(op, field, cm; lifter = lifter) ≈ exact atol = 1e-7
	@test_throws ArgumentError project(op, field, ColoredMatrix(pattern_from_mesh(mesh, 1)); lifter = lifter)
end

@testset "project with periodic and fixed dofs (test functions follow the constraints)" begin
	mesh = proj_mesh()
	op = proj_op(mesh)
	x = mesh.coords[1, :]
	field = [p[1]^2 + p[2] for p in quads(op)]
	# the Galerkin solution in u = P uᵣ + g: Pᵀ M (P uᵣ + g) = Pᵀ b
	energy(u) = integrate(op, (evaluate(op, u) .- field) .^ 2 ./ 2)
	M = ForwardDiff.hessian(energy, zeros(num_nodes(mesh)))
	b = -ForwardDiff.gradient(energy, zeros(num_nodes(mesh)))
	periodic = Periodic(findall(x .== 1.0), findall(x .== 0.0))  # right edge = left edge
	for lifter ∈ (Lifter(num_nodes(mesh), periodic), Lifter(num_nodes(mesh), periodic, Fixed([4], 0.7)))
		P = lifting_matrix(lifter)
		g = lift_from_zeros(lifter, zeros(lifter.size_reduced))
		expected = P * ((transpose(P) * M * P) \ (transpose(P) * (b - M * g))) + g
		@test project(op, field; lifter = lifter) ≈ expected atol = 1e-12
	end
end
