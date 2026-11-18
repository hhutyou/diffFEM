# Package extensions: HyperHessians (element Hessians), Gmsh (mesh input), WriteVTK (output).
# Loading HyperHessians switches every later assembly to hyper-dual numbers: run this file last.

using Test, Tatva, LinearAlgebra, SparseArrays, StaticArrays, ForwardDiff

const μ, λ = 1.0, 4.0
ψ(∇u) = (F = I + ∇u; J = det(F); μ / 2 * (tr(F' * F) - size(F, 1)) - μ * log(J) + λ / 2 * log(J)^2)

using HyperHessians

@testset "HyperHessians: element Hessians with hyper-dual numbers" begin
	@test Base.get_extension(Tatva, :TatvaHyperHessiansExt) !== nothing
	for (mesh, el, d) ∈ ((unit_square(4, 3), Tri3(), 2), (unit_square(3, 3; type = :quad), Quad4(), 2),
		(Mesh([0.0 1.0 0.0 0.0 1.0; 0.0 0.0 1.0 0.0 1.0; 0.0 0.0 0.0 1.0 1.0], [1 2; 2 3; 3 4; 4 5]), Tetrahedron4(), 3))
		op = Operator(mesh, el)
		density(ξ, Ue, Xe) = ψ(grad(op.element, ξ, Ue, Xe))
		asm = ElementAssembler(op, d)
		U = 0.02 .* sin.(3 .* mesh.coords[1:d, :]) .* mesh.coords[2:2, :]
		data = Tatva._assembly_data(asm, U, (mesh.coords,))
		f, x = Tatva._element_function(asm, density, data..., 1)
		v1, g1, H1 = Tatva._value_gradient_hessian(f, x)  # hyper-duals
		v2, g2, H2 = Tatva._value_gradient_hessian_nested(f, x)
		@test v1 ≈ v2 && g1 ≈ g2 && H1 ≈ H2
		E_global(u) = integrate(op, ψ.(grad(op, reshape(u, d, :))))
		_, g, K = assemble(asm, density, U, mesh.coords)
		@test Matrix(K) ≈ ForwardDiff.hessian(E_global, vec(U)) atol = 1e-10
	end
	# an enclosing ForwardDiff derivative (shape sensitivity) still goes through nested duals
	mesh = unit_square(4, 4)
	op = Operator(mesh, Tri3())
	density(ξ, Ue, Xe) = ψ(grad(op.element, ξ, Ue, Xe))
	asm = ElementAssembler(op, 2)
	U = 0.02 .* sin.(3 .* mesh.coords) .* mesh.coords[2:2, :]
	w = collect(1.0:nnz(asm.pattern))  # weighted: the plain sum of a translation-invariant Hessian is zero
	weighted_K(s) = dot(w, nonzeros(assemble_hessian(asm, density, U, s .* mesh.coords)))
	h = 1e-6
	@test ForwardDiff.derivative(weighted_K, 1.0) ≈ (weighted_K(1 + h) - weighted_K(1 - h)) / 2h rtol = 1e-6
end

using Gmsh: Gmsh, gmsh

# a 2 × 1 rectangle meshed by Gmsh, with physical groups "left" (a boundary line) and "plate"
function rectangle_msh(path; order = 1, quads = false)
	Gmsh.initialize(["-v", "0"]; finalize_atexit = false)
	gmsh.model.add("rectangle")
	s = gmsh.model.occ.addRectangle(0, 0, 0, 2, 1)
	gmsh.model.occ.synchronize()
	lines = [t for (_, t) in gmsh.model.getBoundary([(2, s)], true, false)]
	left = only(l for l in lines if all(abs.(gmsh.model.getBoundingBox(1, l)[[1, 4]]) .< 1e-5))  # x = 0 (boxes are padded)
	gmsh.model.addPhysicalGroup(1, [left], -1, "left")
	gmsh.model.addPhysicalGroup(2, [s], -1, "plate")
	gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.3)
	quads && gmsh.model.mesh.setRecombine(2, s)
	gmsh.model.mesh.generate(2)
	gmsh.model.mesh.setOrder(order)
	gmsh.write(path)
	gmsh.finalize()
	return path
end

@testset "Gmsh: mesh input" begin
	dir = mktempdir()
	for (order, quads, E) ∈ ((1, false, Tri3), (2, false, Tri6), (1, true, Quad4))
		m = read_gmsh(rectangle_msh(joinpath(dir, "r$order$quads.msh"); order, quads))
		@test m.element isa E
		@test spatial_dim(m.mesh) == 2
		op = Operator(m.mesh, m.element)
		@test integrate(op, 1.0) ≈ 2.0  # the area
		@test sort(collect(keys(m.groups))) == ["left", "plate"]
		left = m.groups["left"]
		@test left.dim == 1 && isempty(left.elements)
		@test all(abs.(m.mesh.coords[1, left.nodes]) .< 1e-12)
		@test length(left.nodes) == count(abs.(m.mesh.coords[1, :]) .< 1e-12)
		@test m.groups["plate"].elements == 1:num_elements(m.mesh)
		# a patch test on the read mesh: linear fields have constant gradients
		u = 3 .* m.mesh.coords[1, :] .- m.mesh.coords[2, :]
		@test all(g -> g ≈ SVector(3.0, -1.0), grad(op, u))
	end
	# a tetrahedral box, read from the current Gmsh model (built with the Gmsh API)
	Gmsh.initialize(["-v", "0"]; finalize_atexit = false)
	gmsh.model.add("box")
	gmsh.model.occ.addBox(0, 0, 0, 1, 2, 3)
	gmsh.model.occ.synchronize()
	gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.6)
	gmsh.model.mesh.generate(3)
	m = read_gmsh()
	gmsh.finalize()
	@test m.element isa Tetrahedron4
	@test integrate(Operator(m.mesh, m.element), 1.0) ≈ 6.0
	@test isempty(m.groups)
	@test_throws ArgumentError read_gmsh(joinpath(dir, "missing.msh"))
end

using WriteVTK

@testset "WriteVTK: result output" begin
	mesh = unit_square(3, 2)
	op = Operator(mesh, Tri3())
	n, nel = num_nodes(mesh), num_elements(mesh)
	path = joinpath(mktempdir(), "result")
	files = write_vtk(path, op; point_data = (u = 0.1 .* mesh.coords, T = mesh.coords[1, :], S = zeros(2, 2, n),
			v = [SVector(1.0, 2.0) for _ in 1:n]), cell_data = (id = collect(1.0:nel),))
	@test files == [path * ".vtu"]
	xml = read(first(files), String)
	for name ∈ ("u", "T", "S", "v", "id")
		@test occursin("Name=\"$name\"", xml)
	end
	tag(name) = match(Regex("<DataArray[^>]*Name=\"$name\"[^>]*>"), xml).match
	@test occursin("NumberOfComponents=\"3\"", tag("u"))  # 2D vectors padded to 3 components
	@test occursin("NumberOfComponents=\"3\"", tag("v"))
	@test occursin("NumberOfComponents=\"4\"", tag("S"))
	@test_throws DimensionMismatch write_vtk(path, op; point_data = (u = zeros(n + 1),))
	# time series through WriteVTK's own API
	pvd = paraview_collection(joinpath(dirname(path), "series"))
	for (k, t) ∈ enumerate((0.0, 0.5))
		vtk = vtk_grid(joinpath(dirname(path), "step$k"), op)
		vtk["T"] = t .* mesh.coords[1, :]
		pvd[t] = vtk
	end
	@test isfile(first(close(pvd)))
	@test_throws ArgumentError write_vtk(path, Mesh(rand(2, 4), [1 2; 2 3; 3 4]), Tri6())
end
