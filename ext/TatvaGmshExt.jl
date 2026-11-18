# Mesh input with Gmsh.jl, loaded with `using Gmsh` (see `Tatva.read_gmsh`).

module TatvaGmshExt

using Tatva
using Tatva: Element
using Gmsh: Gmsh, gmsh

# Gmsh element type codes → Tatva elements (the node orders agree)
const ELEMENTS = Dict(1 => Line2(), 8 => Line3(), 2 => Tri3(), 9 => Tri6(), 3 => Quad4(), 16 => Quad8(),
	4 => Tetrahedron4(), 5 => Hexahedron8())

function Tatva.read_gmsh(filename::AbstractString; dim::Union{Nothing, Integer} = nothing,
	spatial_dim::Union{Nothing, Integer} = nothing)
	isfile(filename) || throw(ArgumentError("no such file: $filename"))
	started = Gmsh.initialize(["-v", "0"]; finalize_atexit = false)
	previous = started ? nothing : gmsh.model.getCurrent()
	try
		gmsh.model.add("Tatva.read_gmsh")
		gmsh.merge(filename)
		return _read_model(dim, spatial_dim; mesh_if_empty = true)
	finally
		if started
			gmsh.finalize()
		else
			gmsh.model.remove()
			gmsh.model.setCurrent(previous)
		end
	end
end

function Tatva.read_gmsh(; dim::Union{Nothing, Integer} = nothing, spatial_dim::Union{Nothing, Integer} = nothing)
	Bool(gmsh.isInitialized()) || throw(ArgumentError("read_gmsh() reads the current Gmsh model: initialize Gmsh first"))
	return _read_model(dim, spatial_dim; mesh_if_empty = false)
end

function _read_model(dim, spatial_dim; mesh_if_empty::Bool)
	d = dim === nothing ? Int(gmsh.model.getDimension()) : Int(dim)
	types, element_tags, element_nodes = gmsh.model.mesh.getElements(d)
	if isempty(types) && mesh_if_empty  # a geometry: mesh it
		gmsh.model.mesh.generate(d)
		types, element_tags, element_nodes = gmsh.model.mesh.getElements(d)
	end
	isempty(types) && throw(ArgumentError("the model has no elements of dimension $d"))
	if length(types) > 1
		names = [first(gmsh.model.mesh.getElementProperties(t)) for t in types]
		throw(ArgumentError("the elements of dimension $d have several types ($(join(names, ", "))); Tatva meshes have one"))
	end
	haskey(ELEMENTS, types[1]) ||
		throw(ArgumentError("unsupported Gmsh element type $(first(gmsh.model.mesh.getElementProperties(types[1])))"))
	element = ELEMENTS[types[1]]
	connectivity = reshape(Int.(element_nodes[1]), nodes_per_element(element), :)

	# keep the nodes of these elements, numbered in the order of their tags
	tags, flat_coords, _ = gmsh.model.mesh.getNodes()
	tags = Int.(tags)
	all_coords = reshape(flat_coords, 3, :)
	used = sort!(unique(vec(connectivity)))
	index = zeros(Int, maximum(tags))  # tag -> mesh node
	index[used] = 1:length(used)
	position = zeros(Int, maximum(tags))  # tag -> column of all_coords
	position[tags] = 1:length(tags)
	coords3 = all_coords[:, position[used]]
	sd = spatial_dim === nothing ? _spatial_dim(coords3, d) : Int(spatial_dim)
	mesh = Mesh(coords3[1:sd, :], index[connectivity])

	element_index = Dict(Int(t) => e for (e, t) in enumerate(element_tags[1]))
	groups = Dict{String, NamedTuple{(:dim, :nodes, :elements), Tuple{Int, Vector{Int}, Vector{Int}}}}()
	for (gdim, gtag) ∈ gmsh.model.getPhysicalGroups()
		name = gmsh.model.getPhysicalName(gdim, gtag)
		node_tags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(gdim, gtag)
		nodes = sort!([index[t] for t in Int.(node_tags) if t <= length(index) && index[t] > 0])
		elements = Int[]
		if gdim == d
			for entity ∈ gmsh.model.getEntitiesForPhysicalGroup(gdim, gtag)
				_, entity_tags, _ = gmsh.model.mesh.getElements(gdim, entity)
				for ts ∈ entity_tags, t ∈ ts
					push!(elements, element_index[Int(t)])
				end
			end
			sort!(elements)
		end
		groups[isempty(name) ? "$gdim/$gtag" : name] = (dim = Int(gdim), nodes = nodes, elements = elements)
	end
	return (mesh = mesh, element = element, groups = groups)
end

# 3, minus the trailing coordinates that are zero everywhere, at least the mesh dimension
function _spatial_dim(coords3, d)
	sd = 3
	while sd > max(d, 1) && all(iszero, view(coords3, sd, :))
		sd -= 1
	end
	return sd
end

end # module
