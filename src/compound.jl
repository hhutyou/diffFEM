#--------------------------------------------------------------------------------
# Compound: several fields packed into one flat vector
#--------------------------------------------------------------------------------
#
# tatva declares compounds as Python classes. Here a compound is a `CompoundLayout`
# object (created with `Compound(...)` or the `@compound` macro) which is callable to build
# `CompoundState`s:
#
#	Solution = Compound(:Solution, :u => field((2, AUTO)), :phi => field((AUTO,)); mesh = mesh)
#	state = Solution()             # zeros (and default factories)
#	state = Solution(arr)          # wrap a flat vector
#	u, phi = state                 # views into state.arr
#	Solution.u[:, nodes]           # flat (1-based) indices of the u-dofs of `nodes`
#
# Field shapes follow the Julia axis convention: the node axis is LAST, so `(2, AUTO)` is a
# 2-component nodal field. The flat layout is identical to tatva's (reversed) one.

"""`AUTO` in the last dimension of a field shape is resolved to the number of mesh nodes."""
const AUTO = -1

"""Error raised for an invalid compound definition."""
struct CompoundError <: Exception
	msg::String
end
Base.showerror(io::IO, e::CompoundError) = print(io, "CompoundError: ", e.msg)
const CompoundStackError = CompoundError

"""Where the values of a field live: [`Local`](@ref), [`Nodal`](@ref) or [`Shared`](@ref)."""
abstract type FieldType end

"""Field local to an MPI rank (no coupling to other ranks); the default."""
struct Local <: FieldType end

"""Field replicated on every rank and owned by rank 0."""
struct Shared <: FieldType end

"""
	Nodal(; node_ids=nothing, stack=true)
	Nodal(node_ids; stack=true)

Field defined on the mesh nodes (`node_ids === nothing`) or on a subset of local nodes.
Full nodal fields of a compound are stacked node by node unless `stack=false`.
"""
struct Nodal <: FieldType
	node_ids::Union{Nothing, Vector{Int}}
	stack::Bool
end
Nodal(; node_ids = nothing, stack::Bool = true) = Nodal(node_ids === nothing ? nothing : _dof_vector(node_ids), stack)
Nodal(node_ids; stack::Bool = true) = Nodal(; node_ids = node_ids, stack = stack)
Base.:(==)(a::Nodal, b::Nodal) = a.node_ids == b.node_ids && a.stack == b.stack

const LOCAL = Local()
const NODAL = Nodal()
const SHARED = Shared()

"""Specification of a field (before its position in the flat vector is known)."""
struct FieldSpec
	shape::Tuple{Vararg{Int}}
	default_factory::Any
	field_type::FieldType
end

"""
	field(shape; default_factory=nothing, field_type=LOCAL)

Declare a field of a [`Compound`](@ref). `shape` may end with [`AUTO`](@ref) (number of mesh
nodes; such fields are `NODAL` by default). `default_factory()` provides the initial value.
"""
function field(shape, default_factory = nothing, field_type::FieldType = LOCAL; kwargs...)
	df = get(kwargs, :default_factory, default_factory)
	ft = get(kwargs, :field_type, field_type)
	shp = shape isa Integer ? (Int(shape),) : Tuple(Int.(shape))
	return FieldSpec(shp, df, ft)
end

"""
	Field

A field placed in the flat vector of a compound: an affine map from field indices to flat
indices (`base_offset + Σ (iₖ - 1) strides[k]`). Plain fields occupy a contiguous range;
stacked fields are strided views into a shared block. Index it to get flat dof indices:
`Solution.u[:, nodes]`.
"""
struct Field{N, R, VI}
	shape::NTuple{N, Int}
	default_factory::Any
	field_type::FieldType
	root::UnitRange{Int}     # contiguous block of the flat vector holding the field
	root_shape::R            # shape of that block
	view_index::VI           # `nothing` for plain fields, indices into the block otherwise
	base_offset::Int         # flat index of field[1, …, 1]
	strides::NTuple{N, Int}
end

Base.size(f::Field) = f.shape
Base.length(f::Field) = prod(f.shape; init = 1)
Base.ndims(::Field{N}) where {N} = N
is_stacked(f::Field) = f.view_index !== nothing
Base.show(io::IO, f::Field) = print(io, "Field(shape=", f.shape, is_stacked(f) ? ", stacked" : "", ")")

_colmajor_strides(shape::Tuple) = isempty(shape) ? () : (1, cumprod(collect(shape))[1:end-1]...)

function _plain_field(spec, offset::Int)
	n = prod(spec.shape; init = 1)
	root = (offset+1):(offset+n)
	return Field(spec.shape, spec.default_factory, spec.field_type, root, spec.shape, nothing, offset + 1,
		_colmajor_strides(spec.shape))
end

"""View of field `f` into the flat vector `arr` (no copy)."""
field_view(f::Field{N, R, Nothing}, arr::AbstractVector) where {N, R} = reshape(view(arr, f.root), f.shape)
field_view(f::Field, arr::AbstractVector) = reshape(view(reshape(view(arr, f.root), f.root_shape), f.view_index...), f.shape)

#--------------------------------------------------------------------------------
# flat indices of a field
#--------------------------------------------------------------------------------

"""
	indices(field, I...) -> Vector{Int}

Flat (1-based) indices of the entries `I...` of `field`, in column-major order of the
selection. With fewer indices than dimensions the given ones address the trailing axes
(e.g. `u[nodes]` ≡ `u[:, nodes]` for a `(n_comp, n_nodes)` field), as in tatva.
"""
function indices(f::Field{N}, I...) where {N}
	length(I) <= N || throw(ArgumentError("too many indices ($(length(I))) for a field of shape $(f.shape)"))
	idx = (ntuple(_ -> Colon(), N - length(I))..., I...)
	per_axis = ntuple(k -> _axis_indices(idx[k], f.shape[k]), N)
	out = Vector{Int}(undef, prod(length, per_axis; init = 1))
	for (n, J) ∈ enumerate(Iterators.product(per_axis...))
		out[n] = f.base_offset + sum(ntuple(k -> (J[k] - 1) * f.strides[k], N); init = 0)
	end
	return out
end
Base.getindex(f::Field, I...) = indices(f, I...)
Base.lastindex(f::Field, d::Integer) = f.shape[d]
Base.lastindex(f::Field) = f.shape[end]
Base.firstindex(f::Field, ::Integer) = 1
Base.firstindex(f::Field) = 1

_axis_indices(::Colon, n::Int) = collect(1:n)
function _axis_indices(i::Integer, n::Int)
	1 <= i <= n || throw(BoundsError(1:n, i))
	return [Int(i)]
end
_axis_indices(mask::AbstractVector{Bool}, n::Int) = (length(mask) == n || throw(BoundsError(1:n, mask)); findall(mask))
function _axis_indices(v::AbstractVector{<:Integer}, n::Int)
	all(i -> 1 <= i <= n, v) || throw(BoundsError(1:n, v))
	return Vector{Int}(v)
end

#--------------------------------------------------------------------------------
# layouts
#--------------------------------------------------------------------------------

"""
	CompoundLayout

The layout of a compound state (field names, shapes and positions in the flat vector);
created by [`Compound`](@ref). Call it to create states, access a field descriptor with
`layout.u` (index it for flat dof indices) and the total size with `layout.size`.
"""
struct CompoundLayout{Name, F <: NamedTuple}
	fields::F
	size::Int
	mesh::Any
	mpi::Base.RefValue{Any}  # MPI context, set by the MPI extension
end

const _RESERVED_NAMES = (:arr, :layout, :at, :size, :fields, :mesh, :mpi, :name)

function Base.getproperty(l::CompoundLayout{Name}, name::Symbol) where {Name}
	name === :fields && return getfield(l, :fields)
	name === :size && return getfield(l, :size)
	name === :mesh && return getfield(l, :mesh)
	name === :mpi && return getfield(l, :mpi)
	name === :name && return Name
	return getfield(getfield(l, :fields), name)
end
Base.propertynames(l::CompoundLayout) = (:fields, :size, :mesh, :name, keys(getfield(l, :fields))...)

function Base.show(io::IO, l::CompoundLayout{Name}) where {Name}
	parts = ["$(n)=$(f.shape)" for (n, f) in pairs(l.fields)]
	print(io, Name, "(", join(parts, ", "), "; size=", l.size, ")")
end

"""
	Compound(name, :field1 => field(...), :field2 => field(...), ...; mesh=nothing, parent=nothing,
	         partition_info=nothing, comm=nothing)

Define a compound layout (tatva's `class Name(Compound)`). Full nodal fields (`AUTO`-sized
or `NODAL`, with more than one present) are stacked node by node into one block placed
first; the other fields follow in declaration order. The field order of the layout (for
iteration/unpacking) is the declaration order, `parent`'s fields first.

With `partition_info` and `comm` (requires `using MPI`) the distributed dof layout is set up
as well, see [`get_layout`](@ref).
"""
function Compound(name::Symbol, specs::Pair{Symbol, FieldSpec}...; mesh = nothing, parent = nothing,
	partition_info = nothing, comm = nothing)
	inherited = parent === nothing ? Pair{Symbol, Any}[] : Pair{Symbol, Any}[n => f for (n, f) in pairs(parent.fields)]
	offset = parent === nothing ? 0 : parent.size
	mesh = mesh === nothing && parent !== nothing ? parent.mesh : mesh

	seen = Set{Symbol}(first.(inherited))
	for (n, _) ∈ specs
		n in _RESERVED_NAMES && throw(CompoundError("Field name '$n' is reserved and cannot be used in a Compound."))
		n in seen && throw(CompoundError("Duplicate field name '$n'."))
		push!(seen, n)
	end

	full_nodal = Tuple{Symbol, FieldSpec}[]
	others = Tuple{Symbol, FieldSpec}[]
	for (n, spec) ∈ specs
		ft = spec.field_type
		if !isempty(spec.shape) && last(spec.shape) == AUTO
			if ft isa Nodal && ft.node_ids !== nothing
				n_items = length(ft.node_ids)
			else
				mesh === nothing && throw(CompoundError("Mesh must be provided to resolve AUTO size for field '$n'."))
				n_items = num_nodes(mesh)
			end
			spec = FieldSpec((Base.front(spec.shape)..., n_items), spec.default_factory, ft isa Nodal ? ft : NODAL)
		end
		any(<(0), spec.shape) && throw(CompoundError("Invalid shape $(spec.shape) for field '$n'."))
		ft = spec.field_type
		if ft isa Nodal && ft.node_ids === nothing && ft.stack
			push!(full_nodal, (n, spec))
		else
			push!(others, (n, spec))
		end
	end
	# a single full nodal field is not stacked (no strided view needed)
	if length(full_nodal) == 1
		others = vcat(full_nodal, others)
		empty!(full_nodal)
	end

	descriptors = Dict{Symbol, Any}()
	if !isempty(full_nodal)
		d, sz = _stacked_descriptors(full_nodal, offset, 1)
		merge!(descriptors, d)
		offset += sz
	end
	for (n, spec) ∈ others
		descriptors[n] = _plain_field(spec, offset)
		offset += prod(spec.shape; init = 1)
	end

	ordered = vcat(inherited, Pair{Symbol, Any}[n => descriptors[n] for (n, _) in specs])
	layout = _make_layout(name, ordered, offset, mesh)
	if partition_info !== nothing && comm !== nothing
		_set_mpi_context!(layout, partition_info, comm)
	end
	return layout
end
Compound(specs::Pair{Symbol, FieldSpec}...; kwargs...) = Compound(:Compound, specs...; kwargs...)

function _make_layout(name::Symbol, ordered::Vector, size::Int, mesh, mpi = Ref{Any}(nothing))
	names = Tuple(first.(ordered))
	fields = NamedTuple{names}(Tuple(last.(ordered)))
	return CompoundLayout{name, typeof(fields)}(fields, size, mesh, mpi)
end

"""
	extend(parent, :field => field(...), ...; name=:Compound, kwargs...)

A compound with `parent`'s fields followed by new ones (tatva's subclassing).
"""
extend(parent::CompoundLayout, specs::Pair{Symbol, FieldSpec}...; name::Symbol = :Compound, kwargs...) =
	Compound(name, specs...; parent = parent, kwargs...)

# Stack fields along their leading axes: the last `nshared` axes must agree and form the
# block's trailing axes, the leading axes of every field are flattened and concatenated.
function _stacked_descriptors(items, offset::Int, nshared::Int)
	isempty(items) && return Dict{Symbol, Any}(), 0
	first_shape = items[1][2].shape
	length(first_shape) >= nshared ||
		throw(CompoundStackError("Field $(items[1][1]) rank $(length(first_shape)) is not compatible with stacking."))
	trailing = first_shape[(end-nshared+1):end]
	placed = []
	extent_total = 0
	for (n, item) ∈ items
		shape = item.shape
		length(shape) >= nshared ||
			throw(CompoundStackError("Field $n rank $(length(shape)) is not compatible with stacking over $nshared shared axes."))
		shape[(end-nshared+1):end] == trailing ||
			throw(CompoundStackError("Field $n shape $shape does not share the trailing axes $trailing of the stack."))
		lead = shape[1:(end-nshared)]
		extent = prod(lead; init = 1)
		push!(placed, (n, item, extent_total, extent, lead))
		extent_total += extent
	end
	root_shape = (extent_total, trailing...)
	total = prod(root_shape)
	root = (offset+1):(offset+total)
	root_strides = _colmajor_strides(root_shape)
	colons = ntuple(_ -> Colon(), length(trailing))
	descriptors = Dict{Symbol, Any}()
	for (n, item, start, extent, lead) ∈ placed
		view_index = isempty(lead) ? (start + 1, colons...) : ((start+1):(start+extent), colons...)
		strides = (_colmajor_strides(lead)..., root_strides[2:end]...)
		descriptors[n] = Field(item.shape, item.default_factory, item.field_type, root, root_shape, view_index,
			root.start + start, strides)
	end
	return descriptors, total
end

"""
	stack_fields(layout, names...; axis=1)

A new layout in which the fields `names` are interleaved in one block: they are
concatenated along `axis` (the axes `1:axis` of each field are flattened) and must agree on
the axes after `axis` of the first field. The block starts where the first of them was; the
other fields that followed are moved behind the block. E.g. with `u`, `v` of shape
`(2, n)` the block is `(4, n)` and holds `[u[:, k]; v[:, k]]` for every `k`.
"""
function stack_fields(layout::CompoundLayout{Name}, names::Symbol...; axis::Integer = 1) where {Name}
	isempty(names) && throw(CompoundError("At least one field name is required."))
	fields = layout.fields
	for n ∈ names
		haskey(fields, n) || throw(CompoundStackError("Unknown field name in stack_fields: $n"))
		is_stacked(fields[n]) && throw(CompoundStackError("Field $n is already stacked."))
	end
	field_names = collect(keys(fields))
	first_idx = minimum(findfirst(==(n), field_names) for n in names)
	base_offset = first(fields[field_names[first_idx]].root) - 1
	nd = length(fields[names[1]].shape)
	1 <= axis <= max(nd, 1) || throw(CompoundStackError("Invalid stacking axis $axis for a field of rank $nd."))
	descriptors, stack_size = _stacked_descriptors([(n, fields[n]) for n in names], base_offset, nd - axis)
	current = base_offset + stack_size
	ordered = Pair{Symbol, Any}[]
	for (i, n) ∈ enumerate(field_names)
		f = fields[n]
		if i < first_idx
			push!(ordered, n => f)
		elseif n in names
			push!(ordered, n => descriptors[n])
		else
			is_stacked(f) && throw(CompoundStackError("Cannot move the stacked field $n behind a new stack."))
			push!(ordered, n => _plain_field(f, current))
			current += length(f)
		end
	end
	return _make_layout(Name, ordered, current, layout.mesh)
end

"""
	@compound Name [mesh=..] [parent=..] begin
		u = field((2, AUTO))
		phi = field((AUTO,); default_factory = () -> ones(n))
	end

Define `Name = Compound(:Name, :u => field(...), ...; kwargs...)`.
"""
macro compound(name, args...)
	isempty(args) && throw(ArgumentError("@compound needs a begin ... end block of field definitions"))
	block = args[end]
	block isa Expr && block.head === :block || throw(ArgumentError("@compound expects a begin ... end block"))
	specs = Any[]
	for ex ∈ block.args
		ex isa LineNumberNode && continue
		(ex isa Expr && ex.head === :(=) && ex.args[1] isa Symbol) ||
			throw(ArgumentError("@compound: expected `name = field(...)`, got $ex"))
		push!(specs, :($(QuoteNode(ex.args[1])) => $(esc(ex.args[2]))))
	end
	kws = Any[]
	for kw ∈ args[1:(end-1)]
		(kw isa Expr && kw.head === :(=)) || throw(ArgumentError("@compound: expected keyword `key=value`, got $kw"))
		push!(kws, Expr(:kw, kw.args[1], esc(kw.args[2])))
	end
	call = Expr(:call, GlobalRef(@__MODULE__, :Compound), Expr(:parameters, kws...), QuoteNode(name), specs...)
	return :($(esc(name)) = $call)
end

#--------------------------------------------------------------------------------
# states
#--------------------------------------------------------------------------------

"""
	CompoundState

A flat vector `arr` interpreted through a [`CompoundLayout`](@ref): `state.u` is a view
of field `u`, `u, phi = state` unpacks all fields, [`set`](@ref) returns an updated copy.
"""
struct CompoundState{L <: CompoundLayout, A <: AbstractVector}
	layout::L
	arr::A
end

function (layout::CompoundLayout)(arr::AbstractVector)
	length(arr) == layout.size || throw(DimensionMismatch("Array size $(length(arr)) does not match expected size $(layout.size)."))
	return CompoundState(layout, arr)
end

function (layout::CompoundLayout)(::Nothing = nothing; kwargs...)
	T = mapreduce(v -> eltype(v), promote_type, values(kwargs); init = Float64)
	arr = zeros(T, layout.size)
	for (n, f) ∈ pairs(layout.fields)
		if haskey(kwargs, n)
			_assign!(field_view(f, arr), kwargs[n])
		elseif f.default_factory !== nothing
			_assign!(field_view(f, arr), f.default_factory())
		end
	end
	return CompoundState(layout, arr)
end

_assign!(dest::AbstractArray, value::Number) = fill!(dest, value)
function _assign!(dest::AbstractArray, value)
	length(value) == length(dest) || throw(DimensionMismatch("cannot set a field of shape $(size(dest)) from $(size(value))"))
	dest .= reshape(value, size(dest))
	return dest
end

function Base.getproperty(s::CompoundState, name::Symbol)
	name === :arr && return getfield(s, :arr)
	name === :layout && return getfield(s, :layout)
	name === :at && return (n -> _CompoundAt(s, Symbol(n)))
	return field_view(getfield(getfield(s, :layout), :fields)[name], getfield(s, :arr))
end
Base.propertynames(s::CompoundState) = (:arr, :layout, keys(getfield(s, :layout).fields)...)

Base.length(s::CompoundState) = length(s.layout.fields)
Base.iterate(s::CompoundState, i::Int = 1) =
	i > length(s.layout.fields) ? nothing : (field_view(s.layout.fields[i], s.arr), i + 1)

function Base.show(io::IO, s::CompoundState{<:CompoundLayout{Name}}) where {Name}
	parts = ["$(n)=$(f.shape)" for (n, f) in pairs(s.layout.fields)]
	print(io, Name, "(", join(parts, ", "), ")")
end

function Base.:+(a::CompoundState, b::CompoundState)
	a.layout.fields == b.layout.fields || throw(CompoundError("cannot add states of different compounds"))
	return CompoundState(a.layout, a.arr + b.arr)
end

Base.:(==)(a::CompoundState, b::CompoundState) = a.layout.fields == b.layout.fields && a.arr == b.arr

"""The flat vector of a state (same as `state.arr`)."""
flatten(s::CompoundState) = s.arr

"""
	set(state, name, value) -> CompoundState

A copy of `state` with field `name` set to `value` (any array of the right length, or a
number). The element type is promoted, so this works with dual numbers.
"""
function set(s::CompoundState, name::Symbol, value)
	f = s.layout.fields[name]
	T = promote_type(eltype(s.arr), eltype(value))
	arr = copyto!(Vector{T}(undef, length(s.arr)), s.arr)
	_assign!(field_view(f, arr), value)
	return CompoundState(s.layout, arr)
end

# `state.at(:u).set(value)`, as in tatva
struct _CompoundAt{S <: CompoundState}
	state::S
	name::Symbol
end
Base.getproperty(a::_CompoundAt, n::Symbol) =
	n === :set ? (value -> set(getfield(a, :state), getfield(a, :name), value)) : getfield(a, n)

"""
	indices(layout, name, I...)

Flat indices of entries `I...` of field `name` of a compound layout.
"""
indices(l::CompoundLayout, name::Symbol, I...) = indices(l.fields[name], I...)
