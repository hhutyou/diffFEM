#--------------------------------------------------------------------------------
# ColoredMatrix and colored (sparse) forward-mode Jacobians
#--------------------------------------------------------------------------------

"""
	ColoredMatrix(pattern::SparseMatrixCSC; colors=nothing)
	ColoredMatrix(indptr, indices, data, shape; colors=nothing)

A sparse matrix in CSR layout (1-based `indptr`/`indices`, values `data` in row-major
nonzero order, as in tatva) together with a column coloring used to compute it as a
Jacobian with [`jacfwd`](@ref). `colors` default to [`distance2_colors`](@ref) of the
pattern.

Convert with `sparse(cm)` (a `SparseMatrixCSC`, e.g. to solve with `\\`), `Matrix(cm)` /
[`to_dense`](@ref), or [`spsolve`](@ref).
"""
struct ColoredMatrix{Tv, Ti <: Integer}
	data::Vector{Tv}
	indptr::Vector{Ti}
	indices::Vector{Ti}
	shape::NTuple{2, Int}
	colors::Vector{Ti}

	function ColoredMatrix{Tv, Ti}(data, indptr, indices, shape, colors) where {Tv, Ti}
		_check_csr(indptr, indices, data, shape)
		length(colors) == shape[2] || throw(DimensionMismatch("one color per column expected"))
		return new{Tv, Ti}(data, indptr, indices, shape, colors)
	end
end

function _check_csr(indptr, indices, data, shape)
	length(indptr) == shape[1] + 1 || throw(DimensionMismatch("indptr must have n_rows + 1 entries"))
	length(indices) == length(data) == indptr[end] - 1 ||
		throw(DimensionMismatch("indices/data must have nnz = indptr[end] - 1 entries"))
	return nothing
end

function ColoredMatrix(indptr::AbstractVector{<:Integer}, indices::AbstractVector{<:Integer}, data::AbstractVector,
	shape::Tuple{Integer, Integer}; colors = nothing)
	shp = (Int(shape[1]), Int(shape[2]))
	_check_csr(indptr, indices, data, shp)
	cols = colors === nothing ? distance2_colors(_pattern_from_csr(indptr, indices, shp[1]; ncols = shp[2])) : colors
	return ColoredMatrix{eltype(data), Int}(Vector(data), Vector{Int}(indptr), Vector{Int}(indices), shp, Vector{Int}(cols))
end

function ColoredMatrix(A::SparseMatrixCSC; colors = nothing)
	B = copy(transpose(A))  # CSC of Aᵀ == CSR of A
	cols = colors === nothing ? distance2_colors(A) : colors
	return ColoredMatrix{eltype(A), Int}(copy(nonzeros(B)), Vector{Int}(B.colptr), Vector{Int}(rowvals(B)), size(A), Vector{Int}(cols))
end

"""Same as `ColoredMatrix(pattern; colors)` (tatva's `ColoredMatrix.from_csr`)."""
from_csr(A::SparseMatrixCSC; colors = nothing) = ColoredMatrix(A; colors = colors)

Base.size(cm::ColoredMatrix) = cm.shape
Base.size(cm::ColoredMatrix, d::Integer) = cm.shape[d]
Base.eltype(::ColoredMatrix{Tv}) where {Tv} = Tv
SparseArrays.nnz(cm::ColoredMatrix) = length(cm.data)
n_colors(cm::ColoredMatrix) = maximum(cm.colors; init = 0)

Base.show(io::IO, cm::ColoredMatrix{Tv}) where {Tv} =
	print(io, "ColoredMatrix{", Tv, "}(", cm.shape[1], "×", cm.shape[2], ", nnz=", nnz(cm), ", n_colors=", n_colors(cm), ")")

"""
	with_data(cm, data)

A `ColoredMatrix` with the same structure and coloring but new values.
"""
with_data(cm::ColoredMatrix, data::AbstractVector) =
	ColoredMatrix{eltype(data), eltype(cm.indptr)}(Vector(data), cm.indptr, cm.indices, cm.shape, cm.colors)

"""Row index of every stored entry (CSR order)."""
function csr_rows(cm::ColoredMatrix)
	rows = Vector{Int}(undef, nnz(cm))
	for i ∈ 1:cm.shape[1], k ∈ cm.indptr[i]:(cm.indptr[i+1]-1)
		rows[k] = i
	end
	return rows
end

"""
	SparseArrays.sparse(cm::ColoredMatrix) -> SparseMatrixCSC
"""
SparseArrays.sparse(cm::ColoredMatrix) =
	copy(transpose(SparseMatrixCSC(cm.shape[2], cm.shape[1], copy(cm.indptr), copy(cm.indices), copy(cm.data))))

"""Dense matrix of a `ColoredMatrix`."""
to_dense(cm::ColoredMatrix) = Matrix(sparse(cm))
Base.Matrix(cm::ColoredMatrix) = to_dense(cm)

"""COO triplets `(rows, cols, values)` of a `ColoredMatrix` (CSR order)."""
to_coo(cm::ColoredMatrix) = (csr_rows(cm), copy(cm.indices), copy(cm.data))

function Base.:*(cm::ColoredMatrix, x::AbstractVector)
	size(cm, 2) == length(x) || throw(DimensionMismatch("matrix has $(size(cm, 2)) columns, vector has length $(length(x))"))
	T = promote_type(eltype(cm), eltype(x))
	y = zeros(T, size(cm, 1))
	@inbounds for i ∈ 1:size(cm, 1)
		acc = zero(T)
		for k ∈ cm.indptr[i]:(cm.indptr[i+1]-1)
			acc += cm.data[k] * x[cm.indices[k]]
		end
		y[i] = acc
	end
	return y
end

Base.:(==)(a::ColoredMatrix, b::ColoredMatrix) =
	a.shape == b.shape && a.indptr == b.indptr && a.indices == b.indices && a.data == b.data && a.colors == b.colors

#--------------------------------------------------------------------------------
# colored forward-mode Jacobian
#--------------------------------------------------------------------------------

"""
	jacfwd(fn, colored_matrix; color_batch_size=nothing) -> (u, args...; kwargs...) -> ColoredMatrix

Sparse Jacobian of `fn(u, args...; kwargs...)` with respect to `u`, using forward-mode AD
with one seed per color of `colored_matrix` (whose pattern must contain the Jacobian's).
`color_batch_size` colors are propagated per evaluation of `fn` as the partials of
ForwardDiff dual numbers (default: all colors in one evaluation).

`fn` must be generic in its number type. To get a sparse Hessian, differentiate a gradient:
`jacfwd(jacrev(energy), cm)` (or use [`sparse_hessian`](@ref)).
"""
function jacfwd(fn, colored_matrix::ColoredMatrix; color_batch_size::Union{Nothing, Integer} = nothing)
	return ColoredJacobian(fn, colored_matrix, _batch_size(colored_matrix, color_batch_size), false)
end

"""
	linearized_jacfwd(fn, colored_matrix; color_batch_size=nothing) -> (u, args...) -> (fn(u), J)

Like [`jacfwd`](@ref) but also returns the primal value `fn(u, args...)`, shared with the
derivative evaluation.
"""
function linearized_jacfwd(fn, colored_matrix::ColoredMatrix; color_batch_size::Union{Nothing, Integer} = nothing)
	return ColoredJacobian(fn, colored_matrix, _batch_size(colored_matrix, color_batch_size), true)
end

"""
	sparse_hessian(energy, colored_matrix; color_batch_size=nothing, coloring=:acyclic)

Sparse Hessian of a scalar `energy(u, args...)` (a `ColoredMatrix` with the structure of
`colored_matrix`): colored forward mode over the (forward-mode) gradient, one gradient
evaluation per color. The Hessian being symmetric, the colors come from a symmetric coloring
of the pattern (SparseMatrixColorings.jl): `:acyclic` (fewest colors, entries recovered by
substitution) or `:star` (entries read directly); `:column` uses the distance-2 coloring of
`colored_matrix` as for any Jacobian (tatva's `jacfwd(jacrev(energy), cm)`). The cost grows
with the number of dofs times the number of colors; for large element-wise energies use
[`ElementAssembler`](@ref).
"""
function sparse_hessian(energy, colored_matrix::ColoredMatrix; color_batch_size::Union{Nothing, Integer} = nothing,
	coloring::Symbol = :acyclic)
	coloring === :column && return jacfwd(jacrev(energy), colored_matrix; color_batch_size = color_batch_size)
	return _symmetric_jacobian(jacrev(energy), colored_matrix, coloring, color_batch_size)
end

"""
Jacobian of `fn` known to be symmetric (a gradient), evaluated with the colors of a symmetric
coloring and decompressed by SparseMatrixColorings.jl; built by [`sparse_hessian`](@ref).
"""
struct SymmetricJacobian{F, Tv, Ti, R}
	fn::F
	colored_matrix::ColoredMatrix{Tv, Ti}
	result::R  # SparseMatrixColorings coloring result
	batch_size::Int
end

function _symmetric_jacobian(fn, cm::ColoredMatrix, coloring::Symbol, color_batch_size)
	n = size(cm, 1)
	size(cm, 2) == n || throw(DimensionMismatch("a Hessian pattern must be square"))
	pattern = SparseMatrixCSC(n, n, cm.indptr, cm.indices, fill(true, nnz(cm)))  # CSR read as CSC: the transpose
	issymmetric(pattern) || throw(ArgumentError("symmetric coloring needs a symmetric pattern (use coloring = :column)"))
	result = _symmetric_coloring(pattern, coloring)
	nc = SMC.ncolors(result)
	B = color_batch_size === nothing ? nc : Int(color_batch_size)
	B >= 1 || throw(ArgumentError("color_batch_size must be positive"))
	return SymmetricJacobian(fn, cm, result, max(min(B, nc), 1))
end

# fewest colors among a few vertex orders
function _symmetric_coloring(pattern::SparseMatrixCSC, coloring::Symbol)
	decompression = coloring === :acyclic ? :substitution : coloring === :star ? :direct :
					throw(ArgumentError("coloring must be :acyclic, :star or :column, got :$coloring"))
	problem = SMC.ColoringProblem(; structure = :symmetric, partition = :column)
	results = [SMC.coloring(pattern, problem, SMC.GreedyColoringAlgorithm(order; decompression, postprocessing = true))
			   for order in (SMC.NaturalOrder(), SMC.LargestFirst(), SMC.SmallestLast())]
	return argmin(SMC.ncolors, results)
end

n_colors(sj::SymmetricJacobian) = SMC.ncolors(sj.result)

function (sj::SymmetricJacobian)(u::AbstractVector, args...; kwargs...)
	cm = sj.colored_matrix
	length(u) == size(cm, 2) || throw(DimensionMismatch("input has length $(length(u)) but the colored matrix has $(size(cm, 2)) columns"))
	colors = SMC.column_colors(sj.result)  # 0: a neutral color, never seeded
	compressed = _compressed_jacobian(sj.fn, u, colors, n_colors(sj), Val(sj.batch_size), args...; kwargs...)
	size(compressed, 1) == size(cm, 1) ||
		throw(DimensionMismatch("fn returned $(size(compressed, 1)) outputs but the colored matrix has $(size(cm, 1)) rows"))
	n = size(cm, 1)
	A = SparseMatrixCSC(n, n, copy(cm.indptr), copy(cm.indices), zeros(eltype(compressed), nnz(cm)))
	SMC.decompress!(A, compressed, sj.result)
	# symmetric pattern and values: the CSC arrays of A are the CSR arrays of the ColoredMatrix
	return ColoredMatrix{eltype(compressed), eltype(cm.indptr)}(nonzeros(A), cm.indptr, cm.indices, cm.shape, cm.colors)
end

function _batch_size(cm::ColoredMatrix, color_batch_size)
	nc = n_colors(cm)
	nc >= 1 || throw(ArgumentError("colored matrix has no colors"))
	B = color_batch_size === nothing ? nc : Int(color_batch_size)
	B >= 1 || throw(ArgumentError("color_batch_size must be positive"))
	return min(B, nc)
end

struct ColoredJacobian{F, Tv, Ti}
	fn::F
	colored_matrix::ColoredMatrix{Tv, Ti}
	batch_size::Int
	with_primal::Bool
	rows::Vector{Int}
	col_colors::Vector{Int}
end
ColoredJacobian(fn, cm::ColoredMatrix, B::Int, with_primal::Bool) =
	ColoredJacobian(fn, cm, B, with_primal, csr_rows(cm), Vector{Int}(cm.colors[cm.indices]))

function (cj::ColoredJacobian)(u::AbstractVector, args...; kwargs...)
	cm = cj.colored_matrix
	length(u) == size(cm, 2) || throw(DimensionMismatch("input has length $(length(u)) but the colored matrix has $(size(cm, 2)) columns"))
	primal, data = _colored_jacobian(cj.fn, u, cm, cj.rows, cj.col_colors, Val(cj.batch_size), args...; kwargs...)
	J = ColoredMatrix{eltype(data), eltype(cm.indptr)}(data, cm.indptr, cm.indices, cm.shape, cm.colors)
	return cj.with_primal ? (primal, J) : J
end

# (fn(x), the nonzeros of the Jacobian in CSR order): column c of the compressed Jacobian is
# the directional derivative of `fn` along the indicator vector of color c; colors are
# propagated B at a time as the partials of dual numbers and read straight into the nonzeros
function _colored_jacobian(fn::F, x::AbstractVector{T}, cm::ColoredMatrix, rows, col_colors, ::Val{B}, args...;
	kwargs...) where {F, T, B}
	nc = n_colors(cm)
	# The tag must depend on the argument types: dual numbers of an enclosing
	# differentiation may enter through `args`, and ForwardDiff orders tags by creation.
	Tg = typeof(ForwardDiff.Tag(_JacobianTag{F, typeof(args), typeof(values(kwargs))}(), T))
	xd = similar(x, ForwardDiff.Dual{Tg, T, B})
	_seed_colors!(xd, x, cm.colors, 1, Val(B))
	y = fn(xd, args...; kwargs...)
	length(y) == size(cm, 1) || throw(DimensionMismatch("fn returned $(length(y)) outputs but the colored matrix has $(size(cm, 1)) rows"))
	primal = _dual_value(Tg, y)
	data = Vector{typeof(ForwardDiff.partials(Tg, first(y), 1))}(undef, length(rows))
	_store_nonzeros!(data, Tg, y, rows, col_colors, 1, min(B, nc))
	for c0 ∈ (1+B):B:nc
		_seed_colors!(xd, x, cm.colors, c0, Val(B))
		y = fn(xd, args...; kwargs...)
		_store_nonzeros!(data, Tg, y, rows, col_colors, c0, min(B, nc - c0 + 1))
	end
	return primal, data
end

struct _JacobianTag{F, A, K} end

# The compressed Jacobian (n_outputs × nc): column c is the directional derivative of `fn`
# along the indicator vector of color c (colors 0 are not seeded).
function _compressed_jacobian(fn::F, x::AbstractVector{T}, colors::AbstractVector{<:Integer}, nc::Int, ::Val{B},
	args...; kwargs...) where {F, T, B}
	Tg = typeof(ForwardDiff.Tag(_JacobianTag{F, typeof(args), typeof(values(kwargs))}(), T))
	xd = similar(x, ForwardDiff.Dual{Tg, T, B})
	_seed_colors!(xd, x, colors, 1, Val(B))
	y = fn(xd, args...; kwargs...)
	compressed = Matrix{typeof(ForwardDiff.partials(Tg, first(y), 1))}(undef, length(y), nc)
	_store_columns!(compressed, Tg, y, 1, min(B, nc))
	for c0 ∈ (1+B):B:nc
		_seed_colors!(xd, x, colors, c0, Val(B))
		y = fn(xd, args...; kwargs...)
		_store_columns!(compressed, Tg, y, c0, min(B, nc - c0 + 1))
	end
	return compressed
end

function _store_columns!(compressed, ::Type{Tg}, y, c0::Int, w::Int) where {Tg}
	@inbounds for k ∈ 1:w, (i, yi) ∈ enumerate(y)
		compressed[i, c0+k-1] = ForwardDiff.partials(Tg, yi, k)
	end
	return compressed
end

function _seed_colors!(xd::AbstractVector{ForwardDiff.Dual{Tg, T, B}}, x, colors, c0::Int, ::Val{B}) where {Tg, T, B}
	@inbounds for i ∈ eachindex(x, colors)
		k = colors[i] - c0 + 1
		xd[i] = ForwardDiff.Dual{Tg}(x[i], ForwardDiff.Partials{B, T}(ntuple(j -> j == k ? one(T) : zero(T), Val(B))))
	end
	return xd
end

# nonzeros whose column has one of the colors c0, …, c0 + w - 1 of the evaluation `y`
function _store_nonzeros!(data, ::Type{Tg}, y, rows, col_colors, c0::Int, w::Int) where {Tg}
	@inbounds for k ∈ eachindex(data, rows, col_colors)
		c = col_colors[k] - c0 + 1
		1 <= c <= w && (data[k] = ForwardDiff.partials(Tg, y[rows[k]], c))
	end
	return data
end

_dual_value(::Type{Tg}, y::AbstractArray) where {Tg} = map(yi -> ForwardDiff.value(Tg, yi), y)
_dual_value(::Type{Tg}, y::Number) where {Tg} = ForwardDiff.value(Tg, y)

#--------------------------------------------------------------------------------
# sparse linear solve (differentiable)
#--------------------------------------------------------------------------------

"""
	spsolve(A, b)

Solve `A x = b` for a sparse `A` (`ColoredMatrix` or `SparseMatrixCSC`) and a vector or
matrix right-hand side. Dual-number entries in `A` or `b` (ForwardDiff) are handled by
implicit differentiation, `ẋ = A⁻¹ (ḃ - Ȧ x)`, so solves can be differentiated through; the
solution and all tangents (also of nested duals) share one factorization of the values.

The sparse LU is UMFPACK's: lower-precision floats and fixed-size integers are solved in
double precision (the solution has their floating-point type), other number types
(`BigFloat`, `BigInt`, …) with a dense LU.
"""
spsolve(cm::ColoredMatrix, b::AbstractVecOrMat) = spsolve(sparse(cm), b)

function spsolve(A::SparseMatrixCSC, b::AbstractVecOrMat)
	size(A, 1) == size(A, 2) == size(b, 1) || throw(DimensionMismatch("spsolve: A is $(size(A)), b is $(size(b))"))
	return _solve(_factorize(A, promote_type(eltype(A), eltype(b))), b)
end

# factorization of A for right-hand sides of the number type T
_factorize(A::SparseMatrixCSC, ::Type) = _plain_factorize(A)
function _factorize(A::SparseMatrixCSC, ::Type{T}) where {T <: ForwardDiff.Dual}
	Tg, N = ForwardDiff.tagtype(T), ForwardDiff.npartials(T)
	values = SparseMatrixCSC(A.m, A.n, A.colptr, A.rowval, map(a -> ForwardDiff.value(Tg, a), A.nzval))
	tangents = eltype(A) <: ForwardDiff.Dual{Tg} ?
			   ntuple(k -> SparseMatrixCSC(A.m, A.n, A.colptr, A.rowval, map(a -> ForwardDiff.partials(Tg, a, k), A.nzval)), Val(N)) :
			   nothing
	return _DualFactorization{Tg, N}(_factorize(values, ForwardDiff.valtype(T)), tangents)
end

_plain_factorize(A::SparseMatrixCSC{<:Union{Float64, ComplexF64}}) = lu(A)
# types whose (dense) LU would not be more precise than Float64 anyway
const _SolvedInDouble = Union{Float16, Float32, Bool, Base.BitInteger}
_plain_factorize(A::SparseMatrixCSC{<:_SolvedInDouble}) = _Widened{float(eltype(A))}(lu(SparseMatrixCSC{Float64}(A)))
_plain_factorize(A::SparseMatrixCSC{<:Complex{<:_SolvedInDouble}}) = _Widened{float(eltype(A))}(lu(SparseMatrixCSC{ComplexF64}(A)))
_plain_factorize(A::SparseMatrixCSC) = lu(Matrix(A))

# a double-precision factorization of a matrix of the (lower-precision) type T
struct _Widened{T, F}
	factorization::F
end
_Widened{T}(F) where {T} = _Widened{T, typeof(F)}(F)

# A = values + Σₖ εₖ tangents[k] (tangents `nothing`: A has no partials of this tag)
struct _DualFactorization{Tg, N, F, M}
	values::F
	tangents::M
end
_DualFactorization{Tg, N}(values, tangents) where {Tg, N} = _DualFactorization{Tg, N, typeof(values), typeof(tangents)}(values, tangents)

_solve(F, b::AbstractVecOrMat) = F \ b

function _solve(F::_Widened{T}, b::AbstractVecOrMat) where {T}
	x = F.factorization \ b
	S = float(promote_type(T, eltype(b)))
	return eltype(x) === S ? x : convert(AbstractArray{S}, x)
end

function _solve(F::_DualFactorization{Tg, N}, b::AbstractVecOrMat) where {Tg, N}
	x = _solve(F.values, map(v -> ForwardDiff.value(Tg, v), b))  # values may be duals themselves
	# all N tangent systems share the values: solve them as one multi-RHS system
	rhs = reduce(hcat, ntuple(k -> reshape(_tangent_rhs(F, b, x, k), size(b, 1), :), Val(N)))
	ẋ = _solve(F.values, rhs)
	m = size(b, 2)
	parts = ntuple(k -> reshape(ẋ[:, (k-1)*m+1:k*m], size(b)), Val(N))
	return map((xi, ps...) -> ForwardDiff.Dual{Tg}(xi, ps...), x, parts...)
end

function _tangent_rhs(F::_DualFactorization{Tg}, b, x, k) where {Tg}
	ḃ = map(v -> ForwardDiff.partials(Tg, v, k), b)
	return F.tangents === nothing ? ḃ : ḃ - F.tangents[k] * x
end
