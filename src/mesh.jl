#--------------------------------------------------------------------------------
# Mesh
#--------------------------------------------------------------------------------

"""
	Mesh(coords, elements)

A finite element mesh.

- `coords`: `(dim, n_nodes)` matrix, one column per node (may hold dual numbers when
  differentiating with respect to the geometry);
- `elements`: `(nodes_per_element, n_elements)` integer matrix of 1-based node indices,
  one column per element.
"""
struct Mesh{C <: AbstractMatrix, E <: AbstractMatrix{<:Integer}}
	coords::C
	elements::E
end
Mesh(coords::AbstractMatrix) = Mesh(coords, zeros(Int, 0, 0))

num_nodes(mesh::Mesh) = size(mesh.coords, 2)
num_elements(mesh::Mesh) = size(mesh.elements, 2)
spatial_dim(mesh::Mesh) = size(mesh.coords, 1)

Base.show(io::IO, mesh::Mesh) =
	print(io, "Mesh(dim=", spatial_dim(mesh), ", n_nodes=", num_nodes(mesh), ", n_elements=", num_elements(mesh), ")")

"""
	set_coords(mesh, new_coords)

A new mesh with the same connectivity but updated node coordinates.
"""
set_coords(mesh::Mesh, new_coords::AbstractMatrix) = Mesh(new_coords, mesh.elements)

"""
	remake(mesh; coords=mesh.coords, elements=mesh.elements)
"""
remake(mesh::Mesh; coords = mesh.coords, elements = mesh.elements) = Mesh(coords, elements)

"""
	PartitionInfo(nodes_local_to_global, n_owned_nodes)

Partitioning of a mesh across MPI ranks: `nodes_local_to_global[i]` is the (1-based)
global index of local node `i`; owned nodes come first, so nodes `1:n_owned_nodes` are
owned and the rest are ghosts.
"""
struct PartitionInfo
	nodes_local_to_global::Vector{Int}
	n_owned_nodes::Int
end
PartitionInfo(l2g::AbstractVector{<:Integer}, n_owned::Integer) = PartitionInfo(Vector{Int}(l2g), Int(n_owned))

#--------------------------------------------------------------------------------
# structured meshes
#--------------------------------------------------------------------------------

"""
	unit_square(n_x, n_y; type=:triangle, dim=2)

Structured mesh of the unit square with `n_x × n_y` cells (`type = :triangle` splits each
cell in two triangles, `:quad` keeps quadrilaterals). `dim=3` embeds it in the plane `z = 0`.
"""
unit_square(n_x::Integer, n_y::Integer; type = :triangle, dim::Integer = 2) =
	rectangle((0.0, 1.0), (0.0, 1.0), n_x, n_y; type = type, dim = dim)

"""
	rectangle((x0, x1), (y0, y1), n_x, n_y; type=:triangle, dim=2)

Structured mesh of a rectangle. Node `k = i (n_y + 1) + j + 1` sits at `(xᵢ, yⱼ)` (the
y-index varies fastest), exactly as in tatva.
"""
function rectangle(x, y, n_x::Integer, n_y::Integer; type = :triangle, dim::Integer = 2)
	t = Symbol(type)
	t in (:triangle, :quad) || throw(ArgumentError("Element type $type not implemented (use :triangle or :quad)"))
	dim in (2, 3) || throw(ArgumentError("dim must be 2 or 3, got $dim"))
	xs = range(Float64(x[1]), Float64(x[2]); length = n_x + 1)
	ys = range(Float64(y[1]), Float64(y[2]); length = n_y + 1)
	n_nodes = (n_x + 1) * (n_y + 1)
	coords = zeros(Float64, dim, n_nodes)
	node_id(i, j) = i * (n_y + 1) + j + 1
	for i ∈ 0:n_x, j ∈ 0:n_y
		k = node_id(i, j)
		coords[1, k] = xs[i+1]
		coords[2, k] = ys[j+1]
	end
	if t === :triangle
		elements = Matrix{Int}(undef, 3, 2 * n_x * n_y)
		e = 0
		for i ∈ 0:(n_x-1), j ∈ 0:(n_y-1)
			n0, n1, n2, n3 = node_id(i, j), node_id(i + 1, j), node_id(i, j + 1), node_id(i + 1, j + 1)
			elements[:, e+1] .= (n0, n1, n3)
			elements[:, e+2] .= (n0, n3, n2)
			e += 2
		end
	else
		elements = Matrix{Int}(undef, 4, n_x * n_y)
		e = 0
		for i ∈ 0:(n_x-1), j ∈ 0:(n_y-1)
			elements[:, e+1] .= (node_id(i, j), node_id(i + 1, j), node_id(i + 1, j + 1), node_id(i, j + 1))
			e += 1
		end
	end
	return Mesh(coords, elements)
end

#--------------------------------------------------------------------------------
# element sizes
#--------------------------------------------------------------------------------

_cross(a, b) = SVector(a[2] * b[3] - a[3] * b[2], a[3] * b[1] - a[1] * b[3], a[1] * b[2] - a[2] * b[1])

"""
	element_circumdiameters(mesh, element=nothing)

Circumdiameter (2 × circumradius) of every element: exact for triangles (`Tri3`, 2D or 3D)
and tetrahedra; for other elements the largest distance between two of its nodes. Without
`element` the shape is guessed from the number of nodes (3: a triangle, 4 in 3D: a
tetrahedron), and a guess that turns out flat (a quadratic line, a quadrilateral embedded in
3D) uses the node distance.
"""
function element_circumdiameters(mesh::Mesh, element::Union{Nothing, Element} = nothing)
	X = mesh.coords
	E = mesh.elements
	dim = size(X, 1)
	npe = size(E, 1)
	element === nothing || nodes_per_element(element) == npe ||
		throw(ArgumentError("the mesh elements have $npe nodes, $(nameof(typeof(element))) has $(nodes_per_element(element))"))
	shape = _diameter_shape(element, npe, dim)
	guessed = element === nothing
	node(e, k) = SVector{dim}(ntuple(i -> X[i, E[k, e]], dim))
	return map(1:num_elements(mesh)) do e
		d = if shape === :triangle
			_triangle_circumdiameter(node(e, 1), node(e, 2), node(e, 3), guessed)
		elseif shape === :tetrahedron
			_tetrahedron_circumdiameter(node(e, 1), node(e, 2), node(e, 3), node(e, 4), guessed)
		else
			nothing
		end
		d === nothing ? sqrt(maximum(sum(abs2, node(e, j) - node(e, i)) for i in 1:npe for j in (i+1):npe)) : d
	end
end

_diameter_shape(::Nothing, npe, dim) = npe == 3 && dim >= 2 ? :triangle : npe == 4 && dim == 3 ? :tetrahedron : :nodes
_diameter_shape(::Tri3, _, _) = :triangle
_diameter_shape(::Tetrahedron4, _, _) = :tetrahedron
_diameter_shape(::Element, _, _) = :nodes

# `nothing` for a flat simplex whose shape was only guessed
function _triangle_circumdiameter(x1, x2, x3, guessed)
	v1, v2, v3 = x2 - x1, x3 - x1, x3 - x2
	a, b, c = norm(v1), norm(v2), norm(v3)
	area2 = length(v1) == 2 ? abs(v1[1] * v2[2] - v1[2] * v2[1]) : norm(_cross(v1, v2))
	guessed && area2 <= sqrt(eps()) * max(a, b, c)^2 && return nothing
	return a * b * c / max(area2, 1e-12)
end

function _tetrahedron_circumdiameter(x1, x2, x3, x4, guessed)
	a, b, c = x2 - x1, x3 - x1, x4 - x1
	bc, ca, ab = _cross(b, c), _cross(c, a), _cross(a, b)
	vol6 = abs(dot(a, bc))
	guessed && vol6 <= sqrt(eps()) * max(norm(a), norm(b), norm(c))^3 && return nothing
	return norm(sum(abs2, a) * bc + sum(abs2, b) * ca + sum(abs2, c) * ab) / max(vol6, 1e-12)
end

"""
	hmin(mesh, element=nothing)
	hmin(op)

Minimum element diameter of the mesh (see `element_circumdiameters`).
"""
hmin(mesh::Mesh, element::Union{Nothing, Element} = nothing) = minimum(element_circumdiameters(mesh, element))
"""
	hmax(mesh, element=nothing)
	hmax(op)

Maximum element diameter of the mesh (see `element_circumdiameters`).
"""
hmax(mesh::Mesh, element::Union{Nothing, Element} = nothing) = maximum(element_circumdiameters(mesh, element))

#--------------------------------------------------------------------------------
# partitioning
#--------------------------------------------------------------------------------

"""
	extract_local_mesh(mesh_global, element_partition, part) -> (local_mesh, PartitionInfo)

Local mesh of partition `part` (`element_partition[e]` is the partition of element `e`,
typically an MPI rank). Nodes are ordered owned-first then ghosts (PETSc's VecGhost
convention); a node is owned by the smallest partition among the elements touching it.
"""
function extract_local_mesh(mesh_global::Mesh, element_partition::AbstractVector{<:Integer}, part::Integer)
	E = mesh_global.elements
	n_nodes = num_nodes(mesh_global)
	length(element_partition) == size(E, 2) || throw(DimensionMismatch("one partition id per element expected"))
	local_elements = E[:, element_partition .== part]
	all_local_nodes = sort!(unique(vec(local_elements)))

	node_owner = fill(maximum(element_partition) + 1, n_nodes)
	for e ∈ axes(E, 2), k ∈ axes(E, 1)
		node_owner[E[k, e]] = min(node_owner[E[k, e]], element_partition[e])
	end
	is_owned = node_owner[all_local_nodes] .== part
	owned = all_local_nodes[is_owned]
	ghosts = all_local_nodes[.!is_owned]
	local_nodes_global = vcat(owned, ghosts)

	g2l = zeros(Int, n_nodes)
	g2l[local_nodes_global] = 1:length(local_nodes_global)
	coords_local = mesh_global.coords[:, local_nodes_global]
	elements_local = g2l[local_elements]
	return Mesh(coords_local, elements_local), PartitionInfo(local_nodes_global, length(owned))
end

#--------------------------------------------------------------------------------
# point location
#--------------------------------------------------------------------------------

"""
	find_containing_polygons(points, polygons) -> Vector{Int}

Index of the (first) polygon containing each 2D point, or `0` when the point is in no
polygon. `points` is a `(2, n_points)` matrix (or a vector of 2-vectors); `polygons` a
`(2, n_vertices, n_polygons)` array (or a vector of `(2, n_vertices)` matrices). Uses ray
casting with a bounding-box rejection; points exactly on an edge count as inside, so a
point on a shared boundary goes to the lowest-index polygon.
"""
function find_containing_polygons(points, polygons)
	pts = _as_points(points)
	polys = _as_polygons(polygons)
	return [something(findfirst(poly -> _point_in_polygon(p, poly), polys), 0) for p in pts]
end

_as_points(points::AbstractMatrix) = [SVector{2}(points[1, k], points[2, k]) for k in axes(points, 2)]
_as_points(points::AbstractVector) = [SVector{2}(p[1], p[2]) for p in points]
_as_polygons(polys::AbstractArray{<:Number, 3}) =
	[[SVector{2}(polys[1, v, k], polys[2, v, k]) for v in axes(polys, 2)] for k in axes(polys, 3)]
_as_polygons(polys::AbstractVector{<:AbstractMatrix}) = [[SVector{2}(P[1, v], P[2, v]) for v in axes(P, 2)] for P in polys]

function _point_in_polygon(p, vertices)
	px, py = p
	xmin, xmax = extrema(v[1] for v in vertices)
	ymin, ymax = extrema(v[2] for v in vertices)
	(xmin <= px <= xmax && ymin <= py <= ymax) || return false
	n = length(vertices)
	crossings = 0
	for k ∈ 1:n
		p1 = vertices[k]
		p2 = vertices[mod1(k + 1, n)]
		dx, dy = p2[1] - p1[1], p2[2] - p1[2]
		cross = dx * (py - p1[2]) - dy * (px - p1[1])
		on_segment = min(p1[1], p2[1]) <= px <= max(p1[1], p2[1]) && min(p1[2], p2[2]) <= py <= max(p1[2], p2[2])
		isapprox(cross, 0.0; atol = 1e-8) && on_segment && return true
		if (p1[2] <= py < p2[2]) || (p2[2] <= py < p1[2])
			x_intersect = dx * (py - p1[2]) / dy + p1[1]
			px < x_intersect && (crossings += 1)
		end
	end
	return isodd(crossings)
end
