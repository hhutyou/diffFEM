#--------------------------------------------------------------------------------
# Mesh input and result output: package extensions on Gmsh.jl and WriteVTK.jl
#--------------------------------------------------------------------------------

"""
	read_gmsh(filename; dim=nothing, spatial_dim=nothing) -> (; mesh, element, groups)
	read_gmsh(; dim=nothing, spatial_dim=nothing)

Read a mesh with [Gmsh.jl](https://github.com/JuliaFEM/Gmsh.jl) (`using Gmsh` loads this
method): a `.msh` file of any version, or a geometry (`.geo`, `.step`, …) that is meshed
first; without `filename`, the current model of an initialized Gmsh session (a mesh built
with the Gmsh API).

The elements of dimension `dim` (default: the model's dimension) must all be of one type;
`element` is the matching Tatva element (`Tri3()`, `Tetrahedron4()`, …; the node orders
agree). Nodes are numbered `1:n` in the order of their Gmsh tags, keeping the nodes of these
elements only; `coords` have `spatial_dim` rows (default: 3, minus trailing coordinates that
are zero for every node, but at least `dim`).

`groups` maps the name of every physical group to `(; dim, nodes, elements)`: its nodes (for
boundary conditions) and, for groups of dimension `dim`, the indices of its elements
(materials, sources). Unnamed groups are called `"\$dim/\$tag"`.

	mesh, element, groups = read_gmsh("plate.msh")
	op = Operator(mesh, element)
	left = groups["left"].nodes
"""
read_gmsh(args...; kwargs...) = throw(ArgumentError("read_gmsh requires `using Gmsh` to load Tatva's Gmsh extension"))

"""
	write_vtk(filename, op; point_data=(;), cell_data=(;), kwargs...) -> files
	write_vtk(filename, mesh, element; point_data=(;), cell_data=(;), kwargs...)

Write the mesh with nodal and per-element fields to a VTK unstructured-grid file (`.vtu`, for
ParaView) with [WriteVTK.jl](https://github.com/JuliaVTK/WriteVTK.jl) (`using WriteVTK`
loads this method). `point_data` and `cell_data` are named tuples (or dictionaries) of arrays
with the node (element) axis last — `(n,)`, `(n_comp, n)`, `(a, b, n)` — or vectors of static
arrays; 2-component vectors get a zero third component, as ParaView expects. Other keywords
go to `WriteVTK.vtk_grid`, which is also extended: `vtk_grid(filename, op)` returns the
WriteVTK file object, e.g. for time series with `paraview_collection`.

	write_vtk("result", op; point_data = (u = U,), cell_data = (damage = d,))
"""
write_vtk(args...; kwargs...) = throw(ArgumentError("write_vtk requires `using WriteVTK` to load Tatva's WriteVTK extension"))
