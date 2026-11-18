#--------------------------------------------------------------------------------
# Operator: map, integrate, differentiate and interpolate fields on a mesh
#--------------------------------------------------------------------------------

"""
	Operator(mesh, element; cache_weights=false)

Finite element operator for `mesh` discretised with `element`. Provides [`evaluate`](@ref),
[`grad`](@ref), [`hess`](@ref) (fields at quadrature points), [`integrate`](@ref),
[`map_over_quads`](@ref)/[`map_over_elements`](@ref), [`interpolate`](@ref) and
[`project`](@ref). Every method is generic in the number type, so energies built from them
can be differentiated with ForwardDiff (and Enzyme), traced for sparsity, and also
differentiated with respect to the mesh coordinates.

With `cache_weights=true` the integration weights `detJ * w_q` are computed once.

Example:

	mesh = unit_square(10, 10)
	op = Operator(mesh, Tri3())
	ψ(∇u) = 0.5 * sum(abs2, ∇u)
	energy(u) = integrate(op, ψ.(grad(op, u)))
"""
struct Operator{E <: Element, M <: Mesh, DIM, W}
	mesh::M
	element::E
	weights::W  # `nothing`, or the cached (n_quad, n_elements) integration weights

	function Operator{E, M, DIM, W}(mesh::M, element::E, weights::W) where {E, M, DIM, W}
		return new{E, M, DIM, W}(mesh, element, weights)
	end
end

function Operator(mesh::Mesh, element::Element; cache_weights::Bool = false)
	_check_mesh_element(mesh, element)
	dim = spatial_dim(mesh)
	op = Operator{typeof(element), typeof(mesh), dim, Nothing}(mesh, element, nothing)
	cache_weights || return op
	w = _compute_integration_weights(op)
	return Operator{typeof(element), typeof(mesh), dim, typeof(w)}(mesh, element, w)
end

function _check_mesh_element(mesh::Mesh, element::Element)
	X, E = mesh.coords, mesh.elements
	ndims(X) == 2 || throw(ArgumentError("Mesh coordinates must be a matrix shaped (dim, n_nodes)."))
	size(X, 2) > 0 || throw(ArgumentError("Mesh must contain at least one node."))
	size(E, 2) > 0 || throw(ArgumentError("Mesh must contain at least one element."))
	eltype(E) <: Integer || throw(ArgumentError("Mesh element connectivity must contain integer indices."))
	size(E, 1) == nodes_per_element(element) || throw(ArgumentError(
		"Mesh elements have $(size(E, 1)) nodes but $(nameof(typeof(element))) has $(nodes_per_element(element))."))
	lo, hi = extrema(E)
	lo >= 1 || throw(ArgumentError("Mesh element connectivity contains node indices < 1."))
	hi <= size(X, 2) || throw(ArgumentError("Mesh element connectivity references nodes outside the mesh coordinates array."))
	return nothing
end

Base.show(io::IO, op::Operator) = print(io, "Operator(", op.mesh, ", ", op.element, op.weights === nothing ? "" : ", cached weights", ")")

num_nodes(op::Operator) = num_nodes(op.mesh)
num_elements(op::Operator) = num_elements(op.mesh)
num_quad_points(op::Operator) = num_quad_points(op.element)
spatial_dim(::Operator{E, M, DIM}) where {E, M, DIM} = DIM
hmin(op::Operator) = hmin(op.mesh, op.element)
hmax(op::Operator) = hmax(op.mesh, op.element)

"""
	remake(op; mesh=op.mesh, element=op.element, cache_weights=op.weights !== nothing)

A new operator with some of its attributes replaced (tatva's `Operator._replace`).
"""
remake(op::Operator; mesh = op.mesh, element = op.element, cache_weights::Bool = op.weights !== nothing) =
	Operator(mesh, element; cache_weights = cache_weights)

#--------------------------------------------------------------------------------
# gathering element data into static arrays
#--------------------------------------------------------------------------------

# A field defined at the nodes, with its (static) value shape `S`.
struct NodalGather{S, A}
	data::A
end
# A quantity defined per element (element axis last), with value shape `S`.
struct ElementGather{S, A}
	data::A
end

NodalGather(u::AbstractVector{<:Number}) = NodalGather{(), typeof(u)}(u)
NodalGather(u::AbstractArray{<:Number}) = NodalGather{Base.front(size(u)), typeof(u)}(u)
NodalGather(u::AbstractVector{<:StaticArray}) = NodalGather{:static, typeof(u)}(u)
NodalGather(u) = throw(ArgumentError("unsupported nodal field of type $(typeof(u))"))

ElementGather(v::AbstractVector) = ElementGather{(), typeof(v)}(v)
ElementGather(v::AbstractArray{<:Number}) = ElementGather{Base.front(size(v)), typeof(v)}(v)

_n_items(g::NodalGather{:static}) = length(g.data)
_n_items(g::Union{NodalGather, ElementGather}) = size(g.data, ndims(g.data))

# u[s + L (node - 1)] for every node of element e → SArray{Tuple{S..., N}}
@generated function (g::NodalGather{S})(conn::AbstractMatrix, e::Integer, ::Val{N}) where {S, N}
	if S === :static
		nodes = [:(@inbounds g.data[conn[$n, e]]) for n in 1:N]
		return :(_stack_nodes(tuple($(nodes...))))
	end
	L = prod(S; init = 1)
	exprs = [:(@inbounds u[$s+$L*(conn[$n, e]-1)]) for n in 1:N for s in 1:L]
	return quote
		u = g.data
		SArray{Tuple{$(S...), $N}}(tuple($(exprs...)))
	end
end

@generated function (g::ElementGather{S})(conn::AbstractMatrix, e::Integer, ::Val{N}) where {S, N}
	S === () && return :(@inbounds g.data[e])
	L = prod(S)
	exprs = [:(@inbounds v[$s+$L*(e-1)]) for s in 1:L]
	return quote
		v = g.data
		SArray{Tuple{$(S...)}}(tuple($(exprs...)))
	end
end

"""Nodal coordinates of element `e` as an `SMatrix{DIM,N}`."""
@inline function element_coords(op::Operator{E, M, DIM}, e::Integer) where {E, M, DIM}
	return NodalGather{(DIM,), typeof(op.mesh.coords)}(op.mesh.coords)(op.mesh.elements, e, _nval(op.element))
end

# Run g(e) -> SVector{NQ,R} for every element and collect into a (NQ, n_elements) matrix.
# The element type is that of the first element (later results are converted to it).
function _collect_quads(g::G, nq::Int, nel::Int) where {G}
	r1 = g(1)
	out = Matrix{eltype(r1)}(undef, nq, nel)
	_store_column!(out, r1, 1)
	for e ∈ 2:nel
		_store_column!(out, g(e), e)
	end
	return out
end
@inline function _store_column!(out, r, e)
	@inbounds for q ∈ eachindex(r)
		out[q, e] = r[q]
	end
	return out
end

function _collect_elements(g::G, nel::Int) where {G}
	r1 = g(1)
	out = Vector{typeof(r1)}(undef, nel)
	out[1] = r1
	for e ∈ 2:nel
		out[e] = g(e)
	end
	return out
end

#--------------------------------------------------------------------------------
# map over elements and quadrature points
#--------------------------------------------------------------------------------

"""
	map_over_quads(op, f; element_quantity=())

Return a function `(values...; kwargs...) -> Matrix` that evaluates
`f(ξ, el_values...; kwargs...)` at every quadrature point `ξ` of every element and returns
the `(n_quad, n_elements)` matrix of results (tatva's `Operator.map`).

Each positional value is a nodal field (gathered to the element as a static array with the
node axis last), except the positions listed in `element_quantity` (1-based), which are
per-element quantities with the element axis last.
"""
map_over_quads(op::Operator, f; element_quantity = ()) = QuadMap{typeof(f), Tuple(element_quantity), typeof(op)}(f, op)

"""
	map_over_elements(op, f; element_quantity=())

Like [`map_over_quads`](@ref), but calls `f(el_values...; kwargs...)` once per element and
returns a vector with one result per element.
"""
map_over_elements(op::Operator, f; element_quantity = ()) = ElementMap{typeof(f), Tuple(element_quantity), typeof(op)}(f, op)

struct QuadMap{F, EQ, O <: Operator}
	f::F
	op::O
end
struct ElementMap{F, EQ, O <: Operator}
	f::F
	op::O
end

_gatherers(::Val{EQ}, values::Tuple) where {EQ} =
	ntuple(i -> (i in EQ ? ElementGather(values[i]) : NodalGather(values[i])), Val(length(values)))

function (m::QuadMap{F, EQ})(values...; kwargs...) where {F, EQ}
	gathers = _gatherers(Val(EQ), values)
	_check_gathers(m.op, gathers)
	return _quad_map(m.op, gathers) do ξ, el_values...
		m.f(ξ, el_values...; kwargs...)
	end
end

function (m::ElementMap{F, EQ})(values...; kwargs...) where {F, EQ}
	gathers = _gatherers(Val(EQ), values)
	_check_gathers(m.op, gathers)
	op = m.op
	vN = _nval(op.element)
	conn = op.mesh.elements
	return _collect_elements(num_elements(op)) do e
		el_values = map(g -> g(conn, e, vN), gathers)
		m.f(el_values...; kwargs...)
	end
end

function _quad_map(f::F, op::Operator, gathers::Tuple) where {F}
	el = op.element
	vN = _nval(el)
	conn = op.mesh.elements
	return _collect_quads(num_quad_points(el), num_elements(op)) do e
		el_values = map(g -> g(conn, e, vN), gathers)
		map(ξ -> f(ξ, el_values...), el.quad_points)
	end
end

# Apply an element kernel `kernel(el, ξ, V, X)` to a nodal field at every quadrature point.
function _field_quad_map(kernel::K, op::Operator, u) where {K}
	return _field_quad_map(kernel, op, NodalGather(u))
end
function _field_quad_map(kernel::K, op::Operator, gather::NodalGather) where {K}
	el = op.element
	vN = _nval(el)
	_check_nodal(op, gather)
	conn = op.mesh.elements
	return _collect_quads(num_quad_points(el), num_elements(op)) do e
		V = gather(conn, e, vN)
		X = element_coords(op, e)
		map(ξ -> kernel(el, ξ, V, X), el.quad_points)
	end
end

function _check_nodal(op::Operator, g::NodalGather)
	_n_items(g) == num_nodes(op) || throw(DimensionMismatch(
		"nodal field has $(_n_items(g)) nodes (node axis last) but the mesh has $(num_nodes(op))"))
	return nothing
end

# the gathers index without bounds checks: validate every argument once per call
_check_gathers(op::Operator, gathers::Tuple) = foreach(g -> _check_gather(op, g), gathers)
_check_gather(op::Operator, g::NodalGather) = _check_nodal(op, g)
function _check_gather(op::Operator, g::ElementGather)
	_n_items(g) == num_elements(op) || throw(DimensionMismatch(
		"per-element quantity has $(_n_items(g)) entries (element axis last) but the mesh has $(num_elements(op)) elements"))
	return nothing
end

#--------------------------------------------------------------------------------
# fields at quadrature points
#--------------------------------------------------------------------------------

"""
	evaluate(op, u) -> Matrix (n_quad, n_elements)

Values of the nodal field `u` at the quadrature points (tatva's `Operator.eval`). Entries
are numbers for a scalar field and `StaticArray`s otherwise.
"""
evaluate(op::Operator, u) = _field_quad_map(interpolate, op, u)

"""
	grad(op, u) -> Matrix (n_quad, n_elements)

Spatial gradient of the nodal field `u` at the quadrature points: `SVector{dim}` entries for
a scalar field, `SMatrix{n_comp,dim}` (with `∇u[i, j] = ∂uᵢ/∂xⱼ`) for a vector field.
"""
grad(op::Operator, u) = _field_quad_map(gradient, op, u)

"""
	hess(op, u) -> Matrix (n_quad, n_elements)

Hessian of the nodal field `u` at the quadrature points (two spatial axes appended).
"""
hess(op::Operator, u) = _field_quad_map(hessian, op, u)

"""
	quads(op)

Physical coordinates of all quadrature points, `Matrix{SVector{dim}}` of size
`(n_quad, n_elements)`; same as `evaluate(op, op.mesh.coords)`.
"""
quads(op::Operator) = evaluate(op, op.mesh.coords)

#--------------------------------------------------------------------------------
# integration
#--------------------------------------------------------------------------------

"""
	get_integration_weights(op) -> Matrix (n_quad, n_elements)

`detJ(ξ_q) * w_q` at every quadrature point of every element.
"""
get_integration_weights(op::Operator) = op.weights === nothing ? _compute_integration_weights(op) : op.weights

function _compute_integration_weights(op::Operator)
	el = op.element
	return _collect_quads(num_quad_points(el), num_elements(op)) do e
		X = element_coords(op, e)
		map((ξ, w) -> last(get_jacobian(el, ξ, X)) * w, el.quad_points, el.quad_weights)
	end
end

@inline function _element_weights(op::Operator, e::Integer)
	el = op.element
	if op.weights === nothing
		X = element_coords(op, e)
		return map((ξ, w) -> last(get_jacobian(el, ξ, X)) * w, el.quad_points, el.quad_weights)
	else
		return SVector{num_quad_points(el)}(ntuple(q -> @inbounds(op.weights[q, e]), Val(num_quad_points(el))))
	end
end

_is_quad_field(op::Operator, x) = x isa AbstractMatrix && size(x) == (num_quad_points(op), num_elements(op))

"""
	integrate(op, x)

Integral over the mesh of `x`, which can be a number (a constant), a field at the
quadrature points (`(n_quad, n_elements)` matrix of numbers or static arrays) or a nodal
field (interpolated to the quadrature points first). Array-valued integrands give
array-valued integrals.
"""
function integrate(op::Operator, x)
	per = integrate_per_element(op, x)
	acc = per[1]
	@inbounds for e ∈ 2:length(per)
		acc += per[e]
	end
	return acc
end

"""
	integrate_per_element(op, x) -> Vector (n_elements)

Integral of `x` over each element (see [`integrate`](@ref)).
"""
integrate_per_element(op::Operator, x::Number) = _integrate_quad_values(op, (q, e) -> x)
integrate_per_element(op::Operator, x::AbstractMatrix) = _integrate_quads(op, _is_quad_field(op, x) ? x : evaluate(op, x))
integrate_per_element(op::Operator, x) = _integrate_quads(op, evaluate(op, x))  # a nodal field

_integrate_quads(op::Operator, Q::AbstractMatrix) = _integrate_quad_values(op, (q, e) -> @inbounds(Q[q, e]))

function _integrate_quad_values(op::Operator, value::V) where {V}
	nq = num_quad_points(op)
	return _collect_elements(num_elements(op)) do e
		w = _element_weights(op, e)
		acc = value(1, e) * w[1]
		for q ∈ 2:nq
			acc += value(q, e) * w[q]
		end
		acc
	end
end

#--------------------------------------------------------------------------------
# interpolation at arbitrary points
#--------------------------------------------------------------------------------

"""
	make_interpolate(op, points) -> f

Locate the statically known `points` (`(dim, n_points)` matrix or vector of points) in the
mesh and return a function `u -> values` interpolating nodal fields there (a vector with
one value per point). Points are mapped to reference coordinates with Newton's method, so
curved/bilinear elements are handled exactly. Throws if a point lies outside the mesh;
points on shared boundaries are assigned to the lowest-index element.
"""
function make_interpolate(op::Operator{E, M, DIM}, points; tol::Real = 1e-8) where {E, M, DIM}
	el = op.element
	reference_dim(el) == DIM || throw(ArgumentError("interpolation requires elements of the mesh dimension"))
	pts = _as_point_list(Val(DIM), points)
	located = _locate_points(op, pts, tol)
	any(p -> p[1] == 0, located) && error("Some points are outside the mesh, revise the points")
	elements = first.(located)
	ξs = last.(located)
	vN = _nval(el)
	conn = op.mesh.elements
	return function interpolate_fn(u)
		g = NodalGather(u)
		_check_nodal(op, g)
		return [interpolate(el, ξs[k], g(conn, elements[k], vN), element_coords(op, elements[k])) for k in eachindex(elements)]
	end
end

"""
	interpolate(op, u, points)

Interpolate the nodal field `u` at physical `points`; `make_interpolate(op, points)(u)`.
"""
interpolate(op::Operator, u, points) = make_interpolate(op, points)(u)

_as_point_list(::Val{DIM}, points::AbstractMatrix) where {DIM} = [SVector{DIM}(ntuple(i -> points[i, k], DIM)) for k in axes(points, 2)]
_as_point_list(::Val{DIM}, points::AbstractVector) where {DIM} = [SVector{DIM}(p...) for p in points]

function _locate_points(op::Operator{E, M, DIM}, pts, tol) where {E, M, DIM}
	el = op.element
	boxes = [_element_box(el, element_coords(op, e)) for e in 1:num_elements(op)]
	grid = _BoxGrid(boxes)
	return map(pts) do p
		for e ∈ _candidates(grid, p)  # in increasing order
			lo, hi = boxes[e]
			all(lo .<= p .<= hi) || continue
			ξ, converged = _inverse_map(el, element_coords(op, e), p)
			if converged && _is_inside_reference(el, ξ, tol)
				return (e, ξ)
			end
		end
		return (0, _reference_centroid(el))
	end
end

# padded bounding box of the element: the convex hull of its Bernstein control points
# contains it, also for curved elements; unbounded if the control points are unknown
function _element_box(el::Element, X::StaticMatrix{D}) where {D}
	C = _geometry_control_points(el, X)
	C === nothing && return (SVector{D}(ntuple(_ -> -Inf, D)), SVector{D}(ntuple(_ -> Inf, D)))
	lo = SVector{D}(ntuple(i -> _float_value(minimum(C[i, :])), D))
	hi = SVector{D}(ntuple(i -> _float_value(maximum(C[i, :])), D))
	pad = 1e-8 * max(maximum(hi - lo), 1.0)
	return (lo .- pad, hi .+ pad)
end
_float_value(x::ForwardDiff.Dual) = _float_value(ForwardDiff.value(x))  # coordinates being differentiated
_float_value(x::Real) = Float64(x)

# Uniform grid over the element boxes (about one element per cell): cell c lists the
# elements whose box overlaps it, `items[start[c]:start[c+1]-1]`, in increasing order.
struct _BoxGrid{D}
	origin::SVector{D, Float64}
	h::SVector{D, Float64}  # cell size; Inf: a single cell in that direction
	n::NTuple{D, Int}
	start::Vector{Int}
	items::Vector{Int}
end

function _BoxGrid(boxes::Vector{<:Tuple{SVector{D}, SVector{D}}}) where {D}
	lo = reduce((a, b) -> min.(a, b), first.(boxes))
	hi = reduce((a, b) -> max.(a, b), last.(boxes))
	origin, h, n = _grid_cells(lo, hi, length(boxes))
	cell_range(x, i) = isinf(h[i]) ? 0 : clamp(floor(Int, (x - origin[i]) / h[i]), 0, n[i] - 1)
	cells(box) = CartesianIndices(ntuple(i -> (cell_range(box[1][i], i):cell_range(box[2][i], i)) .+ 1, D))
	linear = LinearIndices(n)
	counts = zeros(Int, prod(n) + 1)
	for box ∈ boxes, c ∈ cells(box)
		counts[linear[c]+1] += 1
	end
	start = cumsum(counts) .+ 1
	items = Vector{Int}(undef, start[end] - 1)
	next = start[1:(end-1)]
	for (e, box) ∈ enumerate(boxes), c ∈ cells(box)
		items[next[linear[c]]] = e
		next[linear[c]] += 1
	end
	return _BoxGrid{D}(origin, h, n, start, items)
end

# (origin, cell size, cells per direction) for about one element per cell; a single cell if
# the boxes are unbounded (or flat)
function _grid_cells(lo::SVector{D}, hi::SVector{D}, nel::Int) where {D}
	L = hi - lo
	all(isfinite, L) && all(>(0), L) || return zero(lo), SVector{D}(ntuple(_ -> Inf, D)), ntuple(_ -> 1, D)
	h0 = (prod(L) / nel)^(1 / D)
	n = ntuple(i -> clamp(ceil(Int, L[i] / h0), 1, nel), D)
	while prod(n) > 8 * nel  # very elongated boxes
		n = max.(n .÷ 2, 1)
	end
	return lo, L ./ SVector(n), n
end

function _candidates(grid::_BoxGrid{D}, p) where {D}
	c, stride = 1, 1
	for i ∈ 1:D
		t = isinf(grid.h[i]) ? 0.0 : (p[i] - grid.origin[i]) / grid.h[i]
		0 <= t <= grid.n[i] || return view(grid.items, 1:0)  # outside every box
		c += min(floor(Int, t), grid.n[i] - 1) * stride
		stride *= grid.n[i]
	end
	return view(grid.items, grid.start[c]:(grid.start[c+1]-1))
end

# Newton's method for x(ξ) = p
function _inverse_map(el::Element{D, N}, X::SMatrix{D, N}, p::SVector{D}) where {D, N}
	ξ = _reference_centroid(el)
	scale = max(maximum(abs, X), 1.0)
	for _ ∈ 1:50
		r = X * shape_function(el, ξ) - p
		J = X * transpose(shape_function_derivative(el, ξ))  # ∂x/∂ξ
		Δ = J \ r
		ξ -= Δ
		norm(Δ) <= 1e-14 * (1 + norm(ξ)) && return ξ, true
		norm(r) <= 1e-14 * scale && return ξ, true
	end
	r = X * shape_function(el, ξ) - p
	return ξ, norm(r) <= 1e-9 * scale
end

#--------------------------------------------------------------------------------
# L2 projection
#--------------------------------------------------------------------------------

"""
	project(op, field, colored_matrix=nothing; lifter=nothing)

L2-project a field given at the quadrature points onto the nodal finite element space,
`∫ v (u_h - field) = 0`, solving the sparse system assembled with [`jacfwd`](@ref).
If `colored_matrix` is omitted, a scalar mass-matrix pattern is built from the mesh and
vector/tensor fields are projected component by component; a colored matrix with `d`
dofs per node instead solves the coupled system. A [`Lifter`](@ref) restricts the solve
to its free dofs.
"""
function project(op::Operator, field, colored_matrix = nothing; lifter = nothing)
	fn = make_project_function(num_nodes(op); colored_matrix = colored_matrix, elements = op.mesh.elements, lifter = lifter)
	return fn(op, field)
end
