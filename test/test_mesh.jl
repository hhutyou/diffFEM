# Port of tatva/tests/test_mesh.py, plus tests of the mesh generators and utilities.

using Test, Tatva, LinearAlgebra

@testset "find_containing_polygons includes boundary points" begin
	polygons = cat([0.0 1.0 1.0 0.0; 0.0 0.0 1.0 1.0], [1.0 2.0 2.0 1.0; 0.0 0.0 1.0 1.0]; dims = 3)  # (2, 4, 2)
	points = [0.5 1.0 1.5; 0.5 0.5 0.5]  # the middle point is on the shared boundary
	@test find_containing_polygons(points, polygons) == [1, 1, 2]
	@test find_containing_polygons([[3.0, 0.5]], polygons) == [0]  # outside every polygon
	@test find_containing_polygons(points, [polygons[:, :, 1], polygons[:, :, 2]]) == [1, 1, 2]
end

@testset "unit_square numbering matches tatva" begin
	mesh = unit_square(2, 3)
	@test size(mesh.coords) == (2, 12)
	@test size(mesh.elements) == (3, 12)
	# node k = i (n_y + 1) + j + 1 sits at (x_i, y_j)
	for i ∈ 0:2, j ∈ 0:3
		@test mesh.coords[:, i*4+j+1] ≈ [i / 2, j / 3]
	end
	# first cell, as in tatva: [n0, n1, n3] and [n0, n3, n2] (+1)
	@test mesh.elements[:, 1] == [1, 5, 6]
	@test mesh.elements[:, 2] == [1, 6, 2]
	# all triangles counter-clockwise with area 1/12
	op = Operator(mesh, Tri3())
	@test all(get_integration_weights(op) .≈ 1 / 12)

	quad = unit_square(2, 2; type = :quad)
	@test size(quad.elements) == (4, 4)
	@test quad.elements[:, 1] == [1, 4, 5, 2]
	@test integrate(Operator(quad, Quad4()), 1.0) ≈ 1.0

	m3 = unit_square(2, 2; dim = 3)
	@test size(m3.coords) == (3, 9)
	@test all(m3.coords[3, :] .== 0)
	@test_throws ArgumentError unit_square(2, 2; type = :hex)
end

@testset "rectangle" begin
	mesh = rectangle((-1.0, 3.0), (0.0, 2.0), 4, 2; type = "quad")
	@test extrema(mesh.coords[1, :]) == (-1.0, 3.0)
	@test extrema(mesh.coords[2, :]) == (0.0, 2.0)
	@test integrate(Operator(mesh, Quad4()), 1.0) ≈ 8.0
end

@testset "hmin / hmax are circumdiameters" begin
	mesh = unit_square(4, 4)
	# right isosceles triangles with legs 1/4: circumdiameter = hypotenuse
	@test hmin(mesh) ≈ sqrt(2) / 4
	@test hmax(mesh) ≈ sqrt(2) / 4
	# a tetrahedron with orthogonal unit legs has circumdiameter √3
	tet = Mesh([0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0], reshape([1, 2, 3, 4], 4, 1))
	@test hmax(tet) ≈ sqrt(3)
	# quads fall back to the largest node distance (the diagonal)
	@test hmax(unit_square(2, 2; type = :quad)) ≈ sqrt(2) / 2
	# triangle in 3D
	tri3d = Mesh([0.0 1.0 0.0; 0.0 0.0 0.0; 0.0 0.0 1.0], reshape([1, 2, 3], 3, 1))
	@test hmax(tri3d) ≈ sqrt(2)
end

@testset "hmin / hmax: flat guesses and explicit elements" begin
	# four nodes in 3D are not necessarily a tetrahedron, three not necessarily a triangle
	@test hmax(unit_square(1, 1; type = :quad, dim = 3)) ≈ sqrt(2)
	@test hmax(unit_square(2, 2; type = :quad, dim = 3), Quad4()) ≈ sqrt(2) / 2
	line = Mesh([0.0 2.0 1.0; 0.0 0.0 0.0], reshape([1, 2, 3], 3, 1))  # a quadratic line in 2D
	@test hmax(line) ≈ 2.0
	@test hmax(Operator(line, Line3())) ≈ 2.0
	@test hmax(Mesh([0.0 2.0 1.0], reshape([1, 2, 3], 3, 1))) ≈ 2.0  # ... and in 1D
	@test hmin(Operator(unit_square(2, 2), Tri3())) ≈ sqrt(2) / 2
	@test_throws ArgumentError hmax(line, Tri6())
end

@testset "set_coords keeps the connectivity" begin
	mesh = unit_square(2, 2)
	moved = set_coords(mesh, 2 .* mesh.coords)
	@test moved.elements === mesh.elements
	@test integrate(Operator(moved, Tri3()), 1.0) ≈ 4.0
end

@testset "extract_local_mesh: owned-first ordering and ownership" begin
	mesh = unit_square(2, 2)  # 9 nodes, 8 triangles
	partition = [0, 0, 0, 0, 1, 1, 1, 1]  # x < 0.5 → part 0, x > 0.5 → part 1
	local0, info0 = extract_local_mesh(mesh, partition, 0)
	local1, info1 = extract_local_mesh(mesh, partition, 1)
	@test Tatva.num_elements(local0) == 4 && Tatva.num_elements(local1) == 4
	# nodes on x = 0.5 (global 4, 5, 6) are shared and owned by the smaller part
	@test info0.n_owned_nodes == 6
	@test sort(info0.nodes_local_to_global) == 1:6
	@test info1.n_owned_nodes == 3
	@test sort(info1.nodes_local_to_global[1:3]) == 7:9
	@test sort(info1.nodes_local_to_global[4:end]) == 4:6
	# local connectivity maps back to the global one
	g0 = info0.nodes_local_to_global[local0.elements]
	@test g0 == mesh.elements[:, 1:4]
	g1 = info1.nodes_local_to_global[local1.elements]
	@test g1 == mesh.elements[:, 5:8]
	@test local1.coords == mesh.coords[:, info1.nodes_local_to_global]
	# owned nodes of all parts partition the global nodes
	owned = vcat(info0.nodes_local_to_global[1:info0.n_owned_nodes], info1.nodes_local_to_global[1:info1.n_owned_nodes])
	@test sort(owned) == 1:9
end
