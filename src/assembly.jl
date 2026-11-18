#--------------------------------------------------------------------------------
# Element-wise assembly of energies (a Julia addition to tatva's API)
#--------------------------------------------------------------------------------
#
# tatva differentiates the global energy; JAX/XLA makes `jacfwd(jacrev(E))` scale. In Julia
# the global forward-mode gradient costs one pass per chunk of dofs, so for large meshes the
# energy is better differentiated element by element: ForwardDiff on the few dofs of each
# element (the classic "AD of element routines"), scattered into the mesh pattern. The cost
# is linear in the number of elements.

"""
	ElementAssembler(op, n_comp=1; element_quantity=())

Assembler for energies written element-wise,

	E(u, extras...) = Σₑ Σ_q w_q detJ_q density(ξ_q, Uₑ, extrasₑ...),

where `Uₑ` are the nodal values of the field `u` on element `e` (a static array with the
node axis last, as for [`map_over_quads`](@ref)) and `extras` are further nodal fields — the
mesh coordinates for instance — or per-element quantities (positions in `element_quantity`,
1-based among the extras), which are not differentiated. `u` has `n_comp` components per
node (`(n_comp, n_nodes)` array, or a vector when `n_comp == 1`); dofs are numbered
interleaved, `(node - 1) * n_comp + c`, as in [`pattern_from_mesh`](@ref).

	asm = ElementAssembler(op, 2)
	density(ξ, Uₑ, Xₑ) = ψ(grad(op.element, ξ, Uₑ, Xₑ))
	E, g, K = assemble(asm, density, U, mesh.coords)

See [`assemble`](@ref) (and [`assemble!`](@ref) to reuse the output arrays),
[`assemble_energy`](@ref), [`assemble_gradient`](@ref).

With `threaded = true` and Julia started with several threads (`julia -t auto`) the
elements are evaluated in parallel tasks, group by group: the elements of a group share no
node, so their contributions never collide. `density` is then called concurrently and must
not mutate shared state. Sums are then taken in a different order (results agree with the
serial ones to rounding).
"""
struct ElementAssembler{O <: Operator, NC, EQ}
	op::O
	pattern::SparseMatrixCSC{Float64, Int}
	dofs::Matrix{Int}                       # (n_local_dofs, n_elements)
	positions::Matrix{Int}                  # nzval index of every local pair (a, b), column-major, per element
	groups::Vector{Vector{UnitRange{Int}}}  # blocks of elements, no two blocks of a group share a node
	threaded::Bool
end

function ElementAssembler(op::Operator, n_comp::Integer = 1; element_quantity = (), threaded::Bool = false)
	d = Int(n_comp)
	pattern = SparseMatrixCSC{Float64, Int}(pattern_from_mesh(op.mesh, d))
	dofs = _element_dofs(op.mesh.elements, d)
	nloc, nel = size(dofs)
	positions = Matrix{Int}(undef, nloc * nloc, nel)
	rv = rowvals(pattern)
	for e ∈ 1:nel, b ∈ 1:nloc
		r = nzrange(pattern, dofs[b, e])
		for a ∈ 1:nloc
			positions[(b-1)*nloc+a, e] = first(r) + searchsortedfirst(view(rv, r), dofs[a, e]) - 1
		end
	end
	return ElementAssembler{typeof(op), d, Tuple(element_quantity)}(op, pattern, dofs, positions,
		_element_groups(op.mesh.elements), threaded)
end

Base.show(io::IO, asm::ElementAssembler{O, NC}) where {O, NC} =
	print(io, "ElementAssembler(", asm.op, ", n_comp=", NC, ", nnz=", nnz(asm.pattern), asm.threaded ? ", threaded" : "", ")")

# Blocks of `block` consecutive elements (mesh numberings are spatially coherent), colored
# greedily so that no two blocks of a group share a node: the blocks of a group can be
# assembled concurrently, each one in order (cache friendly). `elements` may also hold the
# dofs an element writes to (0: none).
function _element_groups(elements::AbstractMatrix{<:Integer}; block::Int = 32)
	blocks = [b:min(b + block - 1, size(elements, 2)) for b ∈ 1:block:size(elements, 2)]
	incident = [Int[] for _ ∈ 1:maximum(elements; init = 0)]  # node -> blocks
	for (k, b) ∈ enumerate(blocks), n ∈ unique(view(elements, :, b))
		n > 0 && push!(incident[n], k)
	end
	color = zeros(Int, length(blocks))
	taken = Int[]  # taken[c] == k: color c is used by a neighbour of block k
	for (k, b) ∈ enumerate(blocks)
		for n ∈ view(elements, :, b)
			n > 0 || continue
			for j ∈ incident[n]
				color[j] > 0 && (taken[color[j]] = k)
			end
		end
		c = findfirst(!=(k), taken)
		c === nothing && (push!(taken, 0); c = length(taken))
		color[k] = c
	end
	groups = [UnitRange{Int}[] for _ ∈ taken]
	for (k, b) ∈ enumerate(blocks)
		push!(groups[color[k]], b)
	end
	return groups
end

# acc + Σₑ kernel(e) over the elements but the first (evaluated before, to find the number
# types): in order, or with `threaded` group by group in parallel tasks
function _element_sum(kernel::K, asm::ElementAssembler, acc, groups = asm.groups) where {K}
	if asm.threaded && Threads.nthreads() > 1
		for group ∈ groups
			acc += _parallel_sum(kernel, group, zero(acc))
		end
	else
		for e ∈ 2:num_elements(asm.op)
			acc += kernel(e)
		end
	end
	return acc
end

function _parallel_sum(kernel::K, blocks::Vector{UnitRange{Int}}, z) where {K}
	n_elements = sum(length, blocks; init = 0)
	ntasks = min(Threads.nthreads(), length(blocks), cld(n_elements, 512))  # ≥ 512 elements per task
	ntasks <= 1 && return _blocks_sum(kernel, blocks, z)
	tasks = [Threads.@spawn _blocks_sum(kernel, chunk, z) for chunk ∈ Iterators.partition(blocks, cld(length(blocks), ntasks))]
	return sum(t -> fetch(t)::typeof(z), tasks)
end

function _blocks_sum(kernel::K, blocks, acc) where {K}
	for b ∈ blocks, e ∈ b
		e == 1 || (acc += kernel(e))
	end
	return acc
end

# energy of one element as a function of its flattened local dofs
struct _ElementEnergy{D, E, S, W, X}
	density::D
	element::E
	shape::S   # Val((value_shape..., N))
	weights::W
	extras::X
end

@inline function (f::_ElementEnergy{D, E, Val{S}})(x::SVector) where {D, E, S}
	Ue = SArray{Tuple{S...}}(Tuple(x))
	qp = f.element.quad_points
	acc = f.weights[1] * f.density(qp[1], Ue, f.extras...)
	for q ∈ 2:length(qp)
		acc += f.weights[q] * f.density(qp[q], Ue, f.extras...)
	end
	return acc
end

# Value, gradient and Hessian of f at x in one pass of nested ForwardDiff duals (their tags
# keep enclosing derivatives apart). `using HyperHessians` adds a faster method for element
# energies computed in plain floating point (TatvaHyperHessiansExt).
@inline _value_gradient_hessian(f::F, x::SVector) where {F} = _value_gradient_hessian_nested(f, x)

function _value_gradient_hessian_nested(f::F, x::SVector{L, T}) where {F, L, T}
	T1 = typeof(ForwardDiff.Tag(f, T))
	inner = SVector{L}(ntuple(i -> ForwardDiff.Dual{T1}(x[i], ntuple(j -> T(i == j), Val(L))), Val(L)))
	V = eltype(inner)
	T2 = typeof(ForwardDiff.Tag(f, V))
	xx = SVector{L}(ntuple(i -> ForwardDiff.Dual{T2}(inner[i], ntuple(j -> i == j ? one(V) : zero(V), Val(L))), Val(L)))
	y = f(xx)
	v = ForwardDiff.value(T1, ForwardDiff.value(T2, y))
	g = SVector{L}(ntuple(i -> ForwardDiff.partials(T1, ForwardDiff.value(T2, y), i), Val(L)))
	H = SMatrix{L, L}(ntuple(k -> ForwardDiff.partials(T1, ForwardDiff.partials(T2, y, (k - 1) ÷ L + 1), (k - 1) % L + 1), Val(L * L)))
	return v, g, H
end

function _value_gradient(f::F, x::SVector{L, T}) where {F, L, T}
	Tg = typeof(ForwardDiff.Tag(f, T))
	y = f(SVector{L}(ntuple(i -> ForwardDiff.Dual{Tg}(x[i], ntuple(j -> T(i == j), Val(L))), Val(L))))
	return ForwardDiff.value(Tg, y), SVector{L}(ntuple(i -> ForwardDiff.partials(Tg, y, i), Val(L)))
end

# gathered data of a call: the differentiated field, the extras and the element value shape
function _assembly_data(asm::ElementAssembler{O, NC, EQ}, u, extras) where {O, NC, EQ}
	gather = NodalGather(u)
	_check_nodal(asm.op, gather)
	S = _value_shape(gather)
	prod(S; init = 1) == NC ||
		throw(DimensionMismatch("the field has $(prod(S; init = 1)) components per node, the assembler $NC"))
	extra_gathers = _gatherers(Val(EQ), extras)
	_check_gathers(asm.op, extra_gathers)
	return gather, extra_gathers, Val((S..., nodes_per_element(asm.op.element)))
end
_value_shape(::NodalGather{S}) where {S} = S === :static ? throw(ArgumentError("pass the field as an array (node axis last)")) : S

# (element energy function, local dofs) of element e
@inline function _element_function(asm::ElementAssembler, density, gather, extra_gathers, shape, e::Int)
	op = asm.op
	conn = op.mesh.elements
	vN = _nval(op.element)
	Ue = gather(conn, e, vN)
	extras_e = map(g -> g(conn, e, vN), extra_gathers)
	f = _ElementEnergy(density, op.element, shape, _element_weights(op, e), extras_e)
	return f, SVector{length(Ue)}(Tuple(Ue))
end

# dest[idx] += vals, skipping the index 0 (a contribution to a constrained dof)
@inline function _scatter!(dest, idx, vals)
	@inbounds for (i, v) ∈ zip(idx, vals)
		i > 0 && (dest[i] += v)
	end
	return dest
end

#--------------------------------------------------------------------------------
# energy, gradient, Hessian
#--------------------------------------------------------------------------------

"""
	assemble_energy(asm, density, u, extras...)

`E(u) = Σₑ Σ_q w_q detJ_q density(ξ_q, Uₑ, extrasₑ...)` (see [`ElementAssembler`](@ref)).
"""
function assemble_energy(asm::ElementAssembler, density, u, extras...)
	data = _assembly_data(asm, u, extras)
	f1, x1 = _element_function(asm, density, data..., 1)
	return _assemble_energy(asm, density, data, f1(x1))
end

function _assemble_energy(asm::ElementAssembler, density::D, data, E1) where {D}
	return _element_sum(asm, E1) do e
		f, x = _element_function(asm, density, data..., e)
		f(x)
	end
end

"""
	assemble_gradient(asm, density, u, extras...) -> Vector

Gradient of the element-wise energy with respect to `u`, flat with interleaved dofs.
"""
assemble_gradient(asm::ElementAssembler, density, u, extras...) = _assemble_gradient(nothing, asm, density, u, extras)

"""
	assemble_gradient!(g, asm, density, u, extras...) -> g

[`assemble_gradient`](@ref) into the vector `g` (overwritten).
"""
assemble_gradient!(g::AbstractVector, asm::ElementAssembler, density, u, extras...) =
	_assemble_gradient(_check_gradient(g, asm), asm, density, u, extras)

function _assemble_gradient(g, asm::ElementAssembler, density, u, extras)
	data = _assembly_data(asm, u, extras)
	f1, x1 = _element_function(asm, density, data..., 1)
	first_element = _value_gradient(f1, x1)
	g = g === nothing ? zeros(eltype(last(first_element)), size(asm.pattern, 1)) : fill!(g, zero(eltype(g)))
	_assemble_gradient!(g, asm, density, data, first_element)
	return g
end

# fills g at `dofs` (by default the assembler's), returns the energy
function _assemble_gradient!(g, asm::ElementAssembler, density::D, data, first_element, dofs = asm.dofs,
	groups = asm.groups) where {D}
	v1, g1 = first_element
	_scatter!(g, view(dofs, :, 1), g1)
	return _element_sum(asm, v1, groups) do e
		f, x = _element_function(asm, density, data..., e)
		v, ge = _value_gradient(f, x)
		_scatter!(g, view(dofs, :, e), ge)
		v
	end
end

"""
	assemble(asm, density, u, extras...) -> (E, g, K)

Energy, gradient (flat, interleaved dofs) and sparse Hessian (`SparseMatrixCSC` with the
pattern of the mesh) of the element-wise energy, from one nested-dual evaluation per
element. `K` owns its arrays; [`assemble!`](@ref) refills `g` and `K` in later calls.
"""
function assemble(asm::ElementAssembler, density, u, extras...)
	data = _assembly_data(asm, u, extras)
	f1, x1 = _element_function(asm, density, data..., 1)
	first_element = _value_gradient_hessian(f1, x1)
	T = typeof(first(first_element))
	g = zeros(T, size(asm.pattern, 1))
	P = asm.pattern
	K = SparseMatrixCSC(P.m, P.n, copy(P.colptr), copy(P.rowval), zeros(T, nnz(P)))
	E = _assemble!(g, nonzeros(K), asm, density, data, first_element)
	return E, g, K
end

"""
	assemble!(g, K, asm, density, u, extras...) -> E

[`assemble`](@ref) into existing arrays: the gradient `g` and the Hessian `K`, which must
have the sparsity structure of the assembler (the `K` of an earlier `assemble`, or
`similar(asm.pattern)`), are overwritten; the energy is returned. The element types of `g`
and `K` must hold the values (`Float64` for plain evaluations).
"""
function assemble!(g::AbstractVector, K::SparseMatrixCSC, asm::ElementAssembler, density, u, extras...)
	_check_gradient(g, asm)
	_check_hessian(K, asm)
	data = _assembly_data(asm, u, extras)
	f1, x1 = _element_function(asm, density, data..., 1)
	fill!(g, zero(eltype(g)))
	fill!(nonzeros(K), zero(eltype(K)))
	return _assemble!(g, nonzeros(K), asm, density, data, _value_gradient_hessian(f1, x1))
end

# fills g and the Hessian's nonzeros nz at `dofs` and `positions` (by default the
# assembler's), returns the energy
function _assemble!(g, nz, asm::ElementAssembler, density::D, data, first_element, dofs = asm.dofs,
	positions = asm.positions, groups = asm.groups) where {D}
	v1, g1, H1 = first_element
	_scatter!(g, view(dofs, :, 1), g1)
	_scatter!(nz, view(positions, :, 1), H1)
	return _element_sum(asm, v1, groups) do e
		f, x = _element_function(asm, density, data..., e)
		v, ge, He = _value_gradient_hessian(f, x)
		_scatter!(g, view(dofs, :, e), ge)
		_scatter!(nz, view(positions, :, e), He)
		v
	end
end

"""
	assemble_hessian(asm, density, u, extras...) -> SparseMatrixCSC
"""
assemble_hessian(asm::ElementAssembler, density, u, extras...) = last(assemble(asm, density, u, extras...))

function _check_gradient(g::AbstractVector, asm::ElementAssembler)
	length(g) == size(asm.pattern, 1) ||
		throw(DimensionMismatch("gradient vector has length $(length(g)), the assembler $(size(asm.pattern, 1)) dofs"))
	return g
end

function _check_hessian(K::SparseMatrixCSC, asm::ElementAssembler)
	P = asm.pattern
	size(K) == size(P) || throw(DimensionMismatch("Hessian is $(size(K)), the assembler's pattern $(size(P))"))
	K.colptr == P.colptr && K.rowval == P.rowval ||
		throw(ArgumentError("the Hessian must have the sparsity structure of the assembler's pattern"))
	return K
end

#--------------------------------------------------------------------------------
# reducing assembled operators with a lifter
#--------------------------------------------------------------------------------

"""
	lifting_matrix(lifter) -> SparseMatrixCSC

The constant matrix `P` of the affine lift, `lift_from_zeros(l, u) = P u + lift_from_zeros(l, 0)`
(size `(local size, size_reduced)`). Reduced operators are `Pᵀ g` (= [`reduce_adjoint`](@ref))
and `Pᵀ K P` (= [`reduce_hessian`](@ref)).
"""
function lifting_matrix(l::Lifter)
	src = _lift_sources(l)
	src === nothing && return _lifting_matrix_numeric(l)
	return _selection_matrix(src, local_size(l), l.size_reduced)
end

# With Fixed and Periodic constraints each dof of the lift copies at most one reduced dof:
# `src[i]` is the column of the only entry in row i of P (0: an empty row). `nothing` if
# another constraint type is present.
function _lift_sources(l::Lifter)
	src = zeros(Int, l.size)
	src[l.free_dofs] = 1:l.size_reduced
	for c ∈ l.constraints
		_lift_sources!(c, src) || return nothing
	end
	return src
end
_lift_sources!(c::Fixed, src) = (src[c.dofs] .= 0; true)
_lift_sources!(c::_AnyPeriodic, src) = (src[c.dofs] = src[c.master_dofs]; true)
_lift_sources!(::Constraint, _) = false

function _selection_matrix(src, n, m)
	rows = findall(>(0), view(src, 1:n))
	return sparse(rows, src[rows], ones(length(rows)), n, m)
end

function _lifting_matrix_numeric(l::Lifter)
	base = lift_from_zeros(l, zeros(l.size_reduced))
	cols = [sparse(lift_from_zeros(l, [i == k ? 1.0 : 0.0 for i in 1:l.size_reduced]) - base) for k in 1:l.size_reduced]
	return reduce(hcat, cols)
end

"""
	reduce_hessian(lifter, K)

Reduce an assembled full Hessian to the free dofs: `Pᵀ K P` with `P = lifting_matrix(lifter)`
(the Hessian of `u -> E(lift(lifter, u))`).
"""
function reduce_hessian(l::Lifter, K::AbstractMatrix)
	n, m = local_size(l), l.size_reduced
	size(K) == (n, n) || throw(DimensionMismatch("reduce_hessian: K is $(size(K)), the lift has $n dofs"))
	src = _lift_sources(l)
	if src === nothing
		P = _lifting_matrix_numeric(l)
		return _sparse(transpose(P) * K * P)
	end
	rows = zeros(Int, m)  # the row of P holding column k, if there is exactly one
	nrows = 0
	for i ∈ 1:n
		src[i] > 0 || continue
		rows[src[i]] = i
		nrows += 1
	end
	# P selects dofs (Dirichlet conditions only): a submatrix
	nrows == m && all(>(0), rows) && return _sparse(K[rows, rows])
	return _reduce_by_sources(K, src, m)
end

_sparse(A::SparseMatrixCSC) = A  # `sparse` would copy
_sparse(A::AbstractMatrix) = sparse(A)

# Pᵀ K P: entry K[i, j] adds to (src[i], src[j])
function _reduce_by_sources(K::SparseMatrixCSC, src, m)
	rv, nz = rowvals(K), nonzeros(K)
	I, J, V = Vector{Int}(undef, nnz(K)), Vector{Int}(undef, nnz(K)), similar(nz)
	count = 0
	for j ∈ axes(K, 2), k ∈ nzrange(K, j)
		src[j] > 0 && src[rv[k]] > 0 || continue
		count += 1
		I[count], J[count], V[count] = src[rv[k]], src[j], nz[k]
	end
	return sparse(resize!(I, count), resize!(J, count), resize!(V, count), m, m)
end
_reduce_by_sources(K::AbstractMatrix, src, m) = (P = _selection_matrix(src, size(K, 1), m); _sparse(transpose(P) * K * P))
