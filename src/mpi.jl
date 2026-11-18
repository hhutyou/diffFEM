#--------------------------------------------------------------------------------
# MPI parallelism: data structures (the communication is in ext/TatvaMPIExt.jl)
#--------------------------------------------------------------------------------
#
# Load MPI.jl (`using MPI`) to activate the extension. Conventions: MPI ranks are 0-based
# (as in MPI); all dof indices, local and global, are 1-based, and 0 marks "no index".

"""
	LocalLayout

Distributed dof layout on one rank (tatva's `_LocalLayout`):

- `local_to_global`: global (solver) index of every local dof (owned dofs of rank `r` are
  numbered contiguously `offset+1 : offset+n_owned`);
- `offset`, `n_owned`, `n_total` (local dofs incl. ghosts), `n_global`;
- `owned_mask`: whether each local dof is owned by this rank;
- `natural_l2g`: natural global index of every local dof (from the mesh numbering).
"""
struct LocalLayout
	local_to_global::Vector{Int}
	offset::Int
	n_owned::Int
	n_total::Int
	n_global::Int
	owned_mask::Vector{Bool}
	natural_l2g::Vector{Int}
end

"""Routing to/from one neighbouring rank."""
struct NeighborRoute
	rank::Int
	local_send_idx::Vector{Int}
	recv_local_idx::Vector{Int}
	send_size::Int
	recv_size::Int
end

"""Layout and routing of the owned rows of a distributed Hessian."""
struct HessianLayout
	owned_nnz::Int
	owned_ptr::Vector{Int}
	owned_indices::Vector{Int}
	local_send_idx::Vector{Int}
	recv_local_idx::Vector{Int}
	neighbor_data::Vector{NeighborRoute}
end

"""
	ExchangePlan(layout, local_sparsity_pattern=nothing; comm)

Point-to-point communication plan for parallel assembly on a partitioned mesh: ghost dof
values are gathered from their owners ([`make_scatter_fwd_set`](@ref)) and local
contributions are summed into the owners' rows ([`make_scatter_rev_add`](@ref)), also for
sparse Hessians when a local sparsity pattern is given. Requires `using MPI`.
"""
struct ExchangePlan{C}
	comm::C
	rank::Int
	nranks::Int
	layout::LocalLayout
	rstart::Int  # owned global dofs are rstart+1 : rend
	rend::Int
	global_size::Int
	neighbor_dof_data::Vector{NeighborRoute}
	send_dof::Vector{Int}
	recv_dof::Vector{Int}
	hessian_layout::Union{Nothing, HessianLayout}
end

"""
	AllreducePlan(global_size, global_sparsity_pattern=nothing; comm)

Allreduce-based parallel assembly: every rank holds the full (replicated) vector, computes
the contributions of its elements, and an allreduce sums them; each rank keeps its block of
rows (`rstart+1 : rend`). Requires `using MPI`.
"""
struct AllreducePlan{C}
	comm::C
	rank::Int
	nranks::Int
	global_size::Int
	rstart::Int
	rend::Int
	owned_nnz::Int
	owned_nnz_start::Int  # 1-based position of the first owned nonzero in CSR order
	owned_ptr::Vector{Int}
	owned_indices::Vector{Int}
	plan_hessian::Bool
end

local_size(p::Union{ExchangePlan, AllreducePlan}) = p.rend - p.rstart
owned_nnz(p::AllreducePlan) = p.owned_nnz
function owned_nnz(p::ExchangePlan)
	p.hessian_layout === nothing && throw(ArgumentError("Hessian layout not initialized."))
	return p.hessian_layout.owned_nnz
end
owned_csr(p::AllreducePlan) = (p.owned_ptr, p.owned_indices)
function owned_csr(p::ExchangePlan)
	p.hessian_layout === nothing && throw(ArgumentError("Hessian layout not initialized."))
	return (p.hessian_layout.owned_ptr, p.hessian_layout.owned_indices)
end
Base.getproperty(p::Union{ExchangePlan, AllreducePlan}, name::Symbol) =
	name === :local_size ? local_size(p) :
	name === :owned_csr ? owned_csr(p) :
	name === :owned_nnz && p isa ExchangePlan ? owned_nnz(p) : getfield(p, name)

"""Block distribution of `n` dofs over `size` ranks: the 0-based `[start, end)` of `rank`."""
function dof_range(n::Integer, size::Integer, rank::Integer)
	base, rem = divrem(n, size)
	if rank < rem
		start = rank * (base + 1)
		return start, start + base + 1
	end
	start = rank * base + rem
	return start, start + base
end

# functions implemented by the MPI extension
"""
	make_scatter_fwd_set(plan::ExchangePlan) -> (x_owned -> u_local)

Return a function gathering ghost values: owned values `x_owned` → local vector incl. ghosts.
"""
function make_scatter_fwd_set end

"""
	make_scatter_rev_add(plan::ExchangePlan, local_fn; is_hessian=false)

Return `(args...) -> owned` computing `local_fn(args...)` (a local vector, or a
`ColoredMatrix` when `is_hessian`) and summing all ranks' contributions into the owned rows.
"""
function make_scatter_rev_add end

"""
	make_allgather(plan::AllreducePlan) -> (x_owned -> u_full)
"""
function make_allgather end

"""
	make_allreduce_owned(plan::AllreducePlan, local_fn; is_hessian=false)

Return `(args...) -> owned` summing `local_fn(args...)` over all ranks and keeping the owned
rows (for a Hessian: a `ColoredMatrix` of the owned rows).
"""
function make_allreduce_owned end

"""
	adapt_layout(lifter, layout, comm) -> (reduced_layout, lifter)

Augment the local layout for the lifter's constraints (extra ghosts for `PeriodicMPI`
masters), resolve the constraints' local indices and return the layout of the free dofs
together with the resolved lifter.
"""
function adapt_layout end

function _set_mpi_context! end
function _create_dof_layout end

const _DEFAULT_COMM = Ref{Any}(nothing)

function _require_mpi(what)
	_DEFAULT_COMM[] === nothing && throw(ArgumentError("$what requires MPI: run `using MPI` to load Tatva's MPI extension"))
	return nothing
end

"""
	get_layout(compound_layout) -> LocalLayout

Distributed dof layout of a compound created with `partition_info` and `comm`.
"""
function get_layout(l::CompoundLayout)
	ctx = l.mpi[]
	ctx === nothing && throw(CompoundError("Layout not set on Compound."))
	return ctx.layout
end

"""Global (natural) layout information of one compound field."""
struct FieldGlobalInfo
	global_shape::Tuple{Vararg{Int}}
	global_base_offset::Int
	global_strides::Tuple{Vararg{Int}}
	global_subset::Union{Nothing, Vector{Int}}
end

"""
	global_indices(compound_layout)

Natural global dof indices of the fields of a distributed compound:
`global_indices(Sol).u[c, global_node]`. For fields on a node subset, the node index is a
global node id of the subset; a missing id raises a `BoundsError`.
"""
function global_indices(l::CompoundLayout)
	ctx = l.mpi[]
	ctx === nothing && throw(CompoundError("Compound '$(l.name)' is missing a dof layout. Global view requires a complete MPI layout."))
	return GlobalIndicesView(ctx.global_field_info)
end

struct GlobalIndicesView
	info::Dict{Symbol, FieldGlobalInfo}
end
function Base.getproperty(v::GlobalIndicesView, name::Symbol)
	info = getfield(v, :info)
	haskey(info, name) || throw(ArgumentError("no field '$name' or layout not initialized"))
	return GlobalFieldIndices(info[name])
end

"""Index helper: natural global dof indices of a distributed field."""
struct GlobalFieldIndices
	info::FieldGlobalInfo
end
Base.lastindex(g::GlobalFieldIndices, d::Integer) = g.info.global_shape[d]

function Base.getindex(g::GlobalFieldIndices, I...)
	info = g.info
	shape = info.global_shape
	N = length(shape)
	length(I) <= N || throw(ArgumentError("too many indices for a field of global shape $shape"))
	idx = Any[ntuple(_ -> Colon(), N - length(I))..., I...]
	if info.global_subset !== nothing
		idx[N] = _subset_positions(info.global_subset, idx[N])
	end
	per_axis = ntuple(k -> _axis_indices(idx[k], shape[k]), N)
	out = Vector{Int}(undef, prod(length, per_axis; init = 1))
	for (n, J) ∈ enumerate(Iterators.product(per_axis...))
		out[n] = info.global_base_offset + sum(ntuple(k -> (J[k] - 1) * info.global_strides[k], N); init = 0)
	end
	return out
end

# translate global node ids (the node axis of a subset field) to positions in the subset
_subset_positions(subset, ::Colon) = Colon()
function _subset_positions(subset, i::Integer)
	pos = searchsortedfirst(subset, i)
	(pos <= length(subset) && subset[pos] == i) || throw(BoundsError(subset, i))
	return pos
end
function _subset_positions(subset, r::AbstractUnitRange)
	return searchsortedfirst(subset, first(r)):(searchsortedfirst(subset, last(r) + 1)-1)
end
function _subset_positions(subset, r::AbstractRange)
	step(r) == 1 || throw(ArgumentError("Slicing with a step on a global subset is not supported."))
	return _subset_positions(subset, first(r):last(r))
end
_subset_positions(subset, v::AbstractVector{<:Integer}) = [_subset_positions(subset, i) for i in v]

#--------------------------------------------------------------------------------
# PeriodicMPI: constraint on natural global dof indices
#--------------------------------------------------------------------------------

function PeriodicMPI(dofs, master_dofs, layout::LocalLayout; comm = nothing)
	if comm === nothing
		_require_mpi("PeriodicMPI")
		comm = _DEFAULT_COMM[]
	end
	slaves = _dof_vector(dofs)
	masters = _dof_vector(master_dofs)
	length(slaves) == length(masters) || throw(DimensionMismatch("PeriodicMPI: $(length(slaves)) dofs but $(length(masters)) masters"))
	allunique(slaves) || throw(ArgumentError("PeriodicMPI: a dof appears twice in `dofs` (it would have two masters)"))
	lookup = create_g2l(layout.natural_l2g)
	local_idx = lookup(slaves)
	valid = local_idx .> 0
	slave_natural = slaves[valid]
	master_natural = masters[valid]
	# masters of local slaves that are not in the local layout become extra ghosts
	extra = master_natural[lookup(master_natural).==0]
	# masters are resolved against the augmented layout by `adapt_layout`
	return PeriodicMPI(local_idx[valid], zeros(Int, count(valid)), slave_natural, master_natural, extra, comm, uuid4())
end

"""Resolve a constraint's local indices against an (augmented) layout."""
_resolve_indices(c::Constraint, layout::LocalLayout) = c
function _resolve_indices(c::PeriodicMPI, layout::LocalLayout)
	lookup = create_g2l(layout.natural_l2g)
	slaves = lookup(c.slave_natural)
	masters = lookup(c.master_natural)
	if any(==(0), slaves) || any(==(0), masters)
		throw(LifterError("PeriodicMPI: failed to resolve local indices for periodic dofs. " *
						  "Ensure all required masters were added as ghosts to the layout."))
	end
	return PeriodicMPI(slaves, masters, c.slave_natural, c.master_natural, c.extra_ghost_dofs, c.comm, c.id)
end

"""
	global_view(state)

Gather a distributed compound state: `global_view(state).u` is the full field (natural
global numbering), assembled from the owned values of all ranks. Requires `using MPI`.
"""
function global_view end
