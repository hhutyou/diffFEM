# MPI parallelism for Tatva (port of tatva/mpi.py and tatva/compound/mpi.py), loaded with
# `using MPI`. MPI ranks are 0-based; all dof indices (local, natural and solver-global)
# are 1-based and 0 marks "no index".

module TatvaMPIExt

using Tatva
using Tatva: LocalLayout, NeighborRoute, HessianLayout, ExchangePlan, AllreducePlan, FieldGlobalInfo
using Tatva: CompoundLayout, CompoundState, CompoundError, PartitionInfo, Lifter, Constraint, PeriodicMPI
using Tatva: Nodal, Local, Shared, create_g2l, dof_range, with_data, num_nodes, field_view
using Tatva: _resolve_indices, _colmajor_strides, _dof_vector
using MPI
using SparseArrays

function __init__()
	Tatva._DEFAULT_COMM[] = MPI.COMM_WORLD
	return nothing
end

#--------------------------------------------------------------------------------
# small collective helpers
#--------------------------------------------------------------------------------

_allgather_int(x::Integer, comm) = Vector{Int}(MPI.Allgather(Int(x), comm))

function _allgather_vectors(v::AbstractVector{<:Integer}, comm)
	counts = _allgather_int(length(v), comm)
	buf = Vector{Int}(undef, sum(counts))
	MPI.Allgatherv!(Vector{Int}(v), MPI.VBuffer(buf, counts), comm)
	offsets = cumsum(vcat(0, counts))
	return [buf[(offsets[r]+1):offsets[r+1]] for r in eachindex(counts)]
end

function _exchange(send::AbstractVector{T}, recv_size::Integer, nbr::Integer, comm) where {T}
	recv = Vector{T}(undef, recv_size)
	MPI.Sendrecv!(Vector{T}(send), recv, comm; dest = nbr, source = nbr)
	return recv
end

#--------------------------------------------------------------------------------
# dof layouts
#--------------------------------------------------------------------------------

"""
Global (solver) numbering of all dofs: the owned dofs of rank `r` get the contiguous
indices `offset+1 : offset+n_owned` (in local order); ghosts look their index up from the
owner through a directory indexed by the natural global index.
"""
function Tatva._create_dof_layout(natural_dof_map::AbstractVector{<:Integer}, owned_mask::AbstractVector{Bool},
	n_natural_global::Integer, comm)
	rank = MPI.Comm_rank(comm)
	n_owned = count(owned_mask)
	n_per_rank = _allgather_int(n_owned, comm)
	n_global = sum(n_per_rank)
	offset = sum(n_per_rank[1:rank]; init = 0)

	l2g = zeros(Int, length(natural_dof_map))
	owned = findall(owned_mask)
	l2g[owned] = offset .+ (1:n_owned)

	directory = zeros(Int, n_natural_global)
	directory[natural_dof_map[owned]] = l2g[owned]
	MPI.Allreduce!(directory, max, comm)
	ghosts = findall(.!owned_mask)
	l2g[ghosts] = directory[natural_dof_map[ghosts]]

	return LocalLayout(l2g, offset, n_owned, length(natural_dof_map), n_global, Vector{Bool}(owned_mask), Vector{Int}(natural_dof_map))
end

_n_per_item(root_shape) = prod(root_shape[1:(end-1)]; init = 1)

"""Collective set-up of the distributed dof layout of a compound (tatva's `_layout_from_compound`)."""
function _layout_from_compound(cl::CompoundLayout, partition_info::PartitionInfo, comm)
	rank = MPI.Comm_rank(comm)
	l2g_nodes = partition_info.nodes_local_to_global
	natural = zeros(Int, cl.size)
	owned_mask = falses(cl.size)

	local_max = isempty(l2g_nodes) ? 0 : maximum(l2g_nodes)
	n_nodes_global = MPI.Allreduce(local_max, max, comm)

	current = 0  # number of natural global dofs of the fields processed so far
	root_info = Dict{UnitRange{Int}, Tuple{Int, Int, Tuple}}()
	processed = Set{UnitRange{Int}}()
	field_info = Dict{Symbol, FieldGlobalInfo}()

	for (name, f) ∈ pairs(cl.fields)
		root = f.root
		ft = f.field_type
		per_item = _n_per_item(f.root_shape)

		g_subset = nothing
		if ft isa Nodal && ft.node_ids !== nothing
			g_subset = sort!(unique(reduce(vcat, _allgather_vectors(l2g_nodes[ft.node_ids], comm))))
		end

		if !haskey(root_info, root)
			n_items_global = if ft isa Local
				MPI.Allreduce(length(root), +, comm) ÷ per_item
			elseif ft isa Nodal
				g_subset === nothing ? n_nodes_global : length(g_subset)
			elseif ft isa Shared
				length(root) ÷ per_item
			else
				throw(ArgumentError("Unsupported field type: $(typeof(ft))"))
			end
			global_root_shape = (f.root_shape[1:(end-1)]..., n_items_global)
			root_info[root] = (current, n_items_global, _colmajor_strides(global_root_shape))
		end
		root_start, n_items_global, root_strides = root_info[root]

		global_shape = (f.shape[1:(end-1)]..., n_items_global)
		if Tatva.is_stacked(f)
			# a view into a stacked block: rows `view_index[1]` of the (K, n_items) block
			rows = f.view_index[1]
			first_row = rows isa Integer ? rows : first(rows)
			lead = f.shape[1:(end-1)]
			strides = (_colmajor_strides(lead)..., root_strides[2:end]...)
			field_info[name] = FieldGlobalInfo(global_shape, root_start + first_row, strides, g_subset)
		else
			field_info[name] = FieldGlobalInfo(global_shape, root_start + 1, root_strides, g_subset)
		end

		root in processed && continue
		push!(processed, root)
		size_local = length(root)
		size_global = n_items_global * per_item

		if ft isa Local
			rank_offset = MPI.Exscan(size_local, +, comm)
			rank_offset = rank == 0 ? 0 : rank_offset
			natural[root] = current + rank_offset .+ (1:size_local)
			owned_mask[root] .= true
		elseif ft isa Nodal
			if ft.node_ids !== nothing
				subset_global = l2g_nodes[ft.node_ids]
				owned_global = Set(l2g_nodes[1:partition_info.n_owned_nodes])
				owned_mask[root] = repeat([g in owned_global for g in subset_global]; inner = per_item)
				positions = [searchsortedfirst(g_subset, g) for g in subset_global]
				natural[root] = vec([current + (p - 1) * per_item + c for c in 1:per_item, p in positions])
			else
				natural[root] = vec([current + (g - 1) * per_item + c for c in 1:per_item, g in l2g_nodes])
				owned_mask[first(root):(first(root)+partition_info.n_owned_nodes*per_item-1)] .= true
			end
		elseif ft isa Shared
			natural[root] = current .+ (1:size_local)
			owned_mask[root] .= rank == 0
		end
		current += size_global
	end

	layout = Tatva._create_dof_layout(natural, Vector{Bool}(owned_mask), current, comm)
	return layout, field_info
end

function Tatva._set_mpi_context!(cl::CompoundLayout, partition_info::PartitionInfo, comm)
	cl.mesh === nothing && throw(CompoundError("Mesh must be set on the compound to set the MPI layout."))
	layout, info = _layout_from_compound(cl, partition_info, comm)
	cl.mpi[] = (layout = layout, global_field_info = info, comm = comm)
	return cl
end

#--------------------------------------------------------------------------------
# gathering distributed compound states
#--------------------------------------------------------------------------------

struct GlobalDataView{T}
	full::Vector{T}
	info::Dict{Symbol, FieldGlobalInfo}
end

function Base.getproperty(v::GlobalDataView, name::Symbol)
	info = getfield(v, :info)
	haskey(info, name) || throw(ArgumentError("no field '$name'"))
	fi = info[name]
	return reshape(getfield(v, :full)[Tatva.GlobalFieldIndices(fi)[]], fi.global_shape)
end

function Tatva.global_view(state::CompoundState)
	ctx = state.layout.mpi[]
	ctx === nothing && throw(CompoundError("Global view requires a complete MPI layout."))
	layout = ctx.layout
	owned = ifelse.(layout.owned_mask, state.arr, zero(eltype(state.arr)))
	full = zeros(eltype(state.arr), layout.n_global)
	full[layout.natural_l2g] = owned
	MPI.Allreduce!(full, +, ctx.comm)
	return GlobalDataView(full, ctx.global_field_info)
end

#--------------------------------------------------------------------------------
# lifter on a distributed layout
#--------------------------------------------------------------------------------

function Tatva.adapt_layout(lifter::Lifter, layout::LocalLayout, comm)
	rank = MPI.Comm_rank(comm)
	extra = [c.extra_ghost_dofs for c in lifter.constraints if c isa PeriodicMPI]
	local_extra = isempty(extra) ? Int[] : sort!(unique(reduce(vcat, extra)))
	# rebuilding the layout is collective: every rank takes part as soon as one rank needs
	# extra ghosts (ranks without any keep their dofs and numbering)
	full_layout = if MPI.Allreduce(length(local_extra), max, comm) == 0
		layout
	else
		Tatva._create_dof_layout(vcat(layout.natural_l2g, local_extra), vcat(layout.owned_mask, falses(length(local_extra))),
			layout.n_global, comm)
	end

	constraints = map(c -> _resolve_indices(c, full_layout), lifter.constraints)
	resolved = Lifter(full_layout.n_total, constraints...; nb_extra_ghost_dofs = isempty(local_extra) ? nothing : length(local_extra))
	resolved = with_values(resolved, lifter.runtime_values)

	free = resolved.free_dofs
	free_owned = full_layout.owned_mask[free]
	free_l2g = full_layout.local_to_global[free]
	n_owned = count(free_owned)
	n_per_rank = _allgather_int(n_owned, comm)
	n_global = sum(n_per_rank)
	offset = sum(n_per_rank[1:rank]; init = 0)

	l2g = zeros(Int, length(free))
	l2g[free_owned] = offset .+ (1:n_owned)
	directory = zeros(Int, full_layout.n_global)
	directory[free_l2g[free_owned]] = l2g[free_owned]
	MPI.Allreduce!(directory, max, comm)
	l2g[.!free_owned] = directory[free_l2g[.!free_owned]]

	reduced = LocalLayout(l2g, offset, n_owned, length(free), n_global, Vector{Bool}(free_owned), free_l2g)
	return reduced, resolved
end

#--------------------------------------------------------------------------------
# ExchangePlan: point-to-point ghost exchange
#--------------------------------------------------------------------------------

function Tatva.ExchangePlan(layout::LocalLayout, local_sparsity_pattern = nothing; comm)
	rank = MPI.Comm_rank(comm)
	nranks = MPI.Comm_size(comm)
	rstart, rend = layout.offset, layout.offset + layout.n_owned
	ranges = _gather_ranges(rstart, rend, comm)
	l2g = layout.local_to_global

	# discovery: local (ghost) dofs owned by every other rank
	dofs_from = [nbr == rank ? Int[] : findall(g -> ranges[nbr+1][1] < g <= ranges[nbr+1][2], l2g) for nbr in 0:(nranks-1)]
	send_counts = Int32.(length.(dofs_from))
	recv_counts = MPI.Alltoall(MPI.UBuffer(send_counts, 1), comm)
	neighbors = [nbr for nbr in 0:(nranks-1) if nbr != rank && (send_counts[nbr+1] > 0 || recv_counts[nbr+1] > 0)]

	# resolution: tell every neighbour which of its owned dofs we hold as ghosts
	routes = NeighborRoute[]
	for nbr ∈ neighbors
		requested = _exchange(l2g[dofs_from[nbr+1]], recv_counts[nbr+1], nbr, comm)
		push!(routes, NeighborRoute(nbr, dofs_from[nbr+1], requested .- rstart, Int(send_counts[nbr+1]), Int(recv_counts[nbr+1])))
	end
	send_dof = findall(layout.owned_mask)
	recv_dof = l2g[send_dof] .- rstart

	hessian_layout = local_sparsity_pattern === nothing ? nothing :
					 _nnz_routing(layout, local_sparsity_pattern, rstart, rend, ranges, comm)
	return ExchangePlan(comm, rank, nranks, layout, rstart, rend, layout.n_global, routes, send_dof, recv_dof, hessian_layout)
end

function _gather_ranges(rstart, rend, comm)
	flat = MPI.Allgather([rstart, rend], comm)
	return [(flat[2r-1], flat[2r]) for r in 1:(length(flat)÷2)]
end

function _csr_of(pattern::SparseMatrixCSC)
	B = copy(transpose(pattern))
	return B.colptr, rowvals(B)
end
_csr_of(cm::ColoredMatrix) = (cm.indptr, cm.indices)

function _nnz_routing(layout::LocalLayout, pattern, rstart, rend, ranges, comm)
	rank = MPI.Comm_rank(comm)
	nranks = MPI.Comm_size(comm)
	indptr, indices = _csr_of(pattern)
	l2g = layout.local_to_global
	l_row = reduce(vcat, [fill(i, indptr[i+1] - indptr[i]) for i in 1:(length(indptr)-1)]; init = Int[])
	g_row = l2g[l_row]
	g_col = l2g[indices]
	valid = findall((g_row .> 0) .& (g_col .> 0))  # 0: constrained / unknown dofs
	gr, gc = g_row[valid], g_col[valid]

	in_range(nbr) = (ranges[nbr+1][1] .< gr) .& (gr .<= ranges[nbr+1][2])
	send_to = [valid[in_range(nbr)] for nbr in 0:(nranks-1)]
	send_counts = Int32.(length.(send_to))
	recv_counts = MPI.Alltoall(MPI.UBuffer(send_counts, 1), comm)
	neighbors = [nbr for nbr in 0:(nranks-1) if nbr != rank && (send_counts[nbr+1] > 0 || recv_counts[nbr+1] > 0)]

	mine = in_range(rank)
	rows = [gr[mine]]
	cols = [gc[mine]]
	for nbr ∈ neighbors
		m = in_range(nbr)
		push!(rows, _exchange(gr[m], recv_counts[nbr+1], nbr, comm))
		push!(cols, _exchange(gc[m], recv_counts[nbr+1], nbr, comm))
	end
	all_rows, all_cols = reduce(vcat, rows), reduce(vcat, cols)

	# unique structural nonzeros of the owned rows, sorted by (row, col), and where every
	# received entry goes
	pairs = collect(zip(all_rows, all_cols))
	uniq = sort!(unique(pairs))
	position = Dict(p => k for (k, p) in enumerate(uniq))
	inverse = [position[p] for p in pairs]

	n_local_rows = rend - rstart
	counts = zeros(Int, n_local_rows)
	for (r, _) ∈ uniq
		counts[r-rstart] += 1
	end
	owned_ptr = cumsum(vcat(1, counts))
	owned_indices = [c for (_, c) in uniq]

	routes = NeighborRoute[]
	cur = length(rows[1])
	for (k, nbr) ∈ enumerate(neighbors)
		sz = length(rows[k+1])
		push!(routes, NeighborRoute(nbr, send_to[nbr+1], inverse[(cur+1):(cur+sz)], Int(send_counts[nbr+1]), sz))
		cur += sz
	end
	return HessianLayout(length(uniq), owned_ptr, owned_indices, send_to[rank+1], inverse[1:length(rows[1])], routes)
end

function Tatva.make_scatter_fwd_set(plan::ExchangePlan)
	layout = plan.layout
	comm = plan.comm
	return function scatter_fwd(x_owned::AbstractVector)
		u_local = zeros(eltype(x_owned), layout.n_total)
		u_local[plan.send_dof] = x_owned[plan.recv_dof]
		for r ∈ plan.neighbor_dof_data
			# roles swap with respect to assembly: we send what the neighbour holds as ghosts
			vals = _exchange(x_owned[r.recv_local_idx], r.send_size, r.rank, comm)
			u_local[r.local_send_idx] = vals
		end
		return u_local
	end
end

function Tatva.make_scatter_rev_add(plan::ExchangePlan, local_fn; is_hessian::Bool = false)
	comm = plan.comm
	if is_hessian
		hl = plan.hessian_layout
		hl === nothing && throw(ArgumentError("Hessian layout not initialized."))
		return function scatter_rev_hessian(args...; kwargs...)
			result = local_fn(args...; kwargs...)
			result isa ColoredMatrix || throw(ArgumentError("local_fn must return a ColoredMatrix when is_hessian=true."))
			data = result.data
			owned = zeros(eltype(data), hl.owned_nnz)
			_scatter_add!(owned, hl.recv_local_idx, data[hl.local_send_idx])
			for r ∈ hl.neighbor_data
				vals = _exchange(data[r.local_send_idx], r.recv_size, r.rank, comm)
				_scatter_add!(owned, r.recv_local_idx, vals)
			end
			# the owned rows of the global matrix, in global (solver) column numbering
			return ColoredMatrix{eltype(owned), Int}(owned, hl.owned_ptr, hl.owned_indices, (Tatva.local_size(plan), plan.global_size),
				ones(Int, plan.global_size))
		end
	end
	return function scatter_rev(args...; kwargs...)
		data = local_fn(args...; kwargs...)
		owned = zeros(eltype(data), Tatva.local_size(plan))
		_scatter_add!(owned, plan.recv_dof, data[plan.send_dof])
		for r ∈ plan.neighbor_dof_data
			vals = _exchange(data[r.local_send_idx], r.recv_size, r.rank, comm)
			_scatter_add!(owned, r.recv_local_idx, vals)
		end
		return owned
	end
end

function _scatter_add!(dest, idx, vals)
	for (i, v) ∈ zip(idx, vals)
		dest[i] += v
	end
	return dest
end

#--------------------------------------------------------------------------------
# AllreducePlan: replicated vectors, summed contributions
#--------------------------------------------------------------------------------

function Tatva.AllreducePlan(global_size::Integer, global_sparsity_pattern = nothing; comm)
	rank = MPI.Comm_rank(comm)
	nranks = MPI.Comm_size(comm)
	rstart, rend = dof_range(global_size, nranks, rank)
	if global_sparsity_pattern === nothing
		return AllreducePlan(comm, rank, nranks, Int(global_size), rstart, rend, 0, 1, [1], Int[], false)
	end
	indptr, indices = _csr_of(global_sparsity_pattern)
	length(indptr) - 1 == global_size || throw(DimensionMismatch("global_sparsity_pattern shape does not match global_size"))
	nnz_start = indptr[rstart+1]
	owned_nnz = indptr[rend+1] - nnz_start
	owned_ptr = indptr[(rstart+1):(rend+1)] .- nnz_start .+ 1
	owned_indices = indices[nnz_start:(nnz_start+owned_nnz-1)]
	return AllreducePlan(comm, rank, nranks, Int(global_size), rstart, rend, owned_nnz, nnz_start, owned_ptr, owned_indices, true)
end

function Tatva.make_allgather(plan::AllreducePlan)
	flat = MPI.Allgather([plan.rstart, plan.rend], plan.comm)
	counts = [flat[2r] - flat[2r-1] for r in 1:plan.nranks]
	return function allgather(x_owned::AbstractVector)
		u = Vector{eltype(x_owned)}(undef, plan.global_size)
		MPI.Allgatherv!(Vector(x_owned), MPI.VBuffer(u, counts), plan.comm)
		return u
	end
end

function Tatva.make_allreduce_owned(plan::AllreducePlan, local_fn; is_hessian::Bool = false)
	comm = plan.comm
	if is_hessian
		plan.plan_hessian || throw(ArgumentError("AllreducePlan not initialized with a Hessian sparsity pattern."))
		range = plan.owned_nnz_start:(plan.owned_nnz_start+plan.owned_nnz-1)
		return function allreduce_hessian(args...; kwargs...)
			result = local_fn(args...; kwargs...)
			result isa ColoredMatrix || throw(ArgumentError("local_fn must return a ColoredMatrix when is_hessian=true."))
			data = MPI.Allreduce(result.data, +, comm)
			return ColoredMatrix{eltype(data), Int}(data[range], plan.owned_ptr, plan.owned_indices,
				(plan.rend - plan.rstart, plan.global_size), result.colors)
		end
	end
	return function allreduce_gradient(args...; kwargs...)
		data = MPI.Allreduce(local_fn(args...; kwargs...), +, comm)
		return data[(plan.rstart+1):plan.rend]
	end
end

end # module TatvaMPIExt
