# Result output with WriteVTK.jl, loaded with `using WriteVTK` (see `Tatva.write_vtk`).

module TatvaWriteVTKExt

using Tatva
using Tatva: Element
using WriteVTK
using StaticArrays: StaticArray

_cell_type(::Line2) = VTKCellTypes.VTK_LINE
_cell_type(::Line3) = VTKCellTypes.VTK_QUADRATIC_EDGE
_cell_type(::Tri3) = VTKCellTypes.VTK_TRIANGLE
_cell_type(::Tri6) = VTKCellTypes.VTK_QUADRATIC_TRIANGLE
_cell_type(::Quad4) = VTKCellTypes.VTK_QUAD
_cell_type(::Quad8) = VTKCellTypes.VTK_QUADRATIC_QUAD
_cell_type(::Tetrahedron4) = VTKCellTypes.VTK_TETRA
_cell_type(::Hexahedron8) = VTKCellTypes.VTK_HEXAHEDRON
_cell_type(el::Element) = throw(ArgumentError("no VTK cell type for $(nameof(typeof(el)))"))

function WriteVTK.vtk_grid(filename::AbstractString, mesh::Mesh, element::Element; kwargs...)
	nodes_per_element(element) == size(mesh.elements, 1) ||
		throw(ArgumentError("the mesh elements have $(size(mesh.elements, 1)) nodes, $(nameof(typeof(element))) has $(nodes_per_element(element))"))
	type = _cell_type(element)
	cells = [MeshCell(type, mesh.elements[:, e]) for e in axes(mesh.elements, 2)]
	return vtk_grid(filename, mesh.coords, cells; kwargs...)
end
WriteVTK.vtk_grid(filename::AbstractString, op::Operator; kwargs...) = vtk_grid(filename, op.mesh, op.element; kwargs...)

function Tatva.write_vtk(filename::AbstractString, mesh::Mesh, element::Element; point_data = (;), cell_data = (;), kwargs...)
	vtk = vtk_grid(filename, mesh, element; kwargs...)
	for (name, data) ∈ pairs(point_data)
		vtk[string(name), VTKPointData()] = _vtk_array(data, num_nodes(mesh), name)
	end
	for (name, data) ∈ pairs(cell_data)
		vtk[string(name), VTKCellData()] = _vtk_array(data, num_elements(mesh), name)
	end
	return close(vtk)
end
Tatva.write_vtk(filename::AbstractString, op::Operator; kwargs...) = Tatva.write_vtk(filename, op.mesh, op.element; kwargs...)

# a vector, or an (n_components, n) matrix with 2-component vectors padded to 3
function _vtk_array(data::AbstractArray{<:Number}, n::Int, name)
	size(data, ndims(data)) == n || throw(DimensionMismatch("$name has $(size(data, ndims(data))) entries along its last axis, expected $n"))
	ndims(data) == 1 && return data
	values = reshape(data, :, n)
	return size(values, 1) == 2 ? vcat(values, zeros(eltype(values), 1, n)) : values
end
_vtk_array(data::AbstractVector{<:StaticArray}, n::Int, name) =
	_vtk_array(reshape(reduce(vcat, vec.(data)), length(first(data)), :), n, name)

end # module
