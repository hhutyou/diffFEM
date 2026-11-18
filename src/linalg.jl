"""
	Tatva.Linalg

Small dense tensor operations on single element-sized arrays.

In tatva (JAX) this module exists to keep tiny contractions off XLA's `dot`/GEMM path:
a 2×2 Jacobian vmapped over millions of elements is much faster as a broadcast-and-sum
than as a batched matrix product, and `jnp.linalg.inv` (LU + triangular solve) is absurd for
a matrix that can be inverted in closed form.

Julia has no such compiler boundary: the element kernels in `Tatva` use `StaticArrays`,
whose small products and closed-form `inv`/`det` are fully unrolled and allocation free —
the Julia equivalent of "avoid the GEMM path". This module is kept so that code written
against tatva's `linalg` API keeps working, with the same contract:

- [`inv`](@ref) and [`det`](@ref) take **one** matrix (batching is `map`/broadcast's job)
  and use closed forms for 1×1, 2×2 and 3×3;
- [`contract`](@ref) evaluates one of a fixed set of named two-operand contractions,
  written as einsum-style specs, and refuses anything it has no entry for;
- [`einsum`](@ref) is a drop-in that takes the `contract` path for a known spec and falls
  back to a general (loop based) einsum otherwise.

Index semantics follow the spec letters exactly as in numpy, applied to the Julia array's
axes in order.
"""
module Linalg

using LinearAlgebra: LinearAlgebra
using StaticArrays

export contract, einsum

const _SINGLE_MATRIX = "Batch with map/broadcasting rather than passing a stack"

#--------------------------------------------------------------------------------
# inv / det
#--------------------------------------------------------------------------------

"""
	inv(J)

Closed-form inverse of a single 1×1, 2×2 or 3×3 matrix; larger square matrices fall back
to `LinearAlgebra.inv`. Anything that is not a single matrix is rejected.
"""
function inv(J::AbstractArray)
	ndims(J) == 2 || throw(ArgumentError(
		"Tatva.Linalg.inv takes a single matrix, got shape $(size(J)). $_SINGLE_MATRIX",
	))
	d = size(J, 2)
	size(J, 1) == d || return LinearAlgebra.inv(J) # not square: let LinearAlgebra raise
	if d == 1
		return _like(J, 1, 1, (one(eltype(J)) / J[1, 1],))
	elseif d == 2
		a, b, c, e = J[1, 1], J[1, 2], J[2, 1], J[2, 2]
		Δ = a * e - b * c
		# column-major entries of [e -b; -c a] / Δ
		return _like(J, 2, 2, (e / Δ, -c / Δ, -b / Δ, a / Δ))
	elseif d == 3
		m = J
		c00 = m[2, 2] * m[3, 3] - m[2, 3] * m[3, 2]
		c01 = m[2, 3] * m[3, 1] - m[2, 1] * m[3, 3]
		c02 = m[2, 1] * m[3, 2] - m[2, 2] * m[3, 1]
		c10 = m[1, 3] * m[3, 2] - m[1, 2] * m[3, 3]
		c11 = m[1, 1] * m[3, 3] - m[1, 3] * m[3, 1]
		c12 = m[1, 2] * m[3, 1] - m[1, 1] * m[3, 2]
		c20 = m[1, 2] * m[2, 3] - m[1, 3] * m[2, 2]
		c21 = m[1, 3] * m[2, 1] - m[1, 1] * m[2, 3]
		c22 = m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1]
		Δ = m[1, 1] * c00 + m[1, 2] * c01 + m[1, 3] * c02
		# inverse = adjugate / Δ, adjugate[i, j] = cofactor[j, i]; listed column-major
		return _like(J, 3, 3, (c00 / Δ, c01 / Δ, c02 / Δ, c10 / Δ, c11 / Δ, c12 / Δ, c20 / Δ, c21 / Δ, c22 / Δ))
	end
	return LinearAlgebra.inv(J)
end

"""
	det(J)

Closed-form determinant of a single 1×1, 2×2 or 3×3 matrix (a scalar); larger square
matrices fall back to `LinearAlgebra.det`.
"""
function det(J::AbstractArray)
	ndims(J) == 2 || throw(ArgumentError(
		"Tatva.Linalg.det takes a single matrix, got shape $(size(J)). $_SINGLE_MATRIX",
	))
	d = size(J, 2)
	size(J, 1) == d || return LinearAlgebra.det(J)
	d == 1 && return J[1, 1]
	d == 2 && return J[1, 1] * J[2, 2] - J[1, 2] * J[2, 1]
	if d == 3
		m = J
		return m[1, 1] * (m[2, 2] * m[3, 3] - m[2, 3] * m[3, 2]) +
			   m[1, 2] * (m[2, 3] * m[3, 1] - m[2, 1] * m[3, 3]) +
			   m[1, 3] * (m[2, 1] * m[3, 2] - m[2, 2] * m[3, 1])
	end
	return LinearAlgebra.det(J)
end

# build a matrix of the same "kind" as J (static stays static) from column-major entries
_like(::StaticMatrix, m, n, vals) = SMatrix{m, n}(vals)
_like(::AbstractMatrix, m, n, vals) = reshape(collect(vals), m, n)

#--------------------------------------------------------------------------------
# tensordot along one axis
#--------------------------------------------------------------------------------

"""
	_dot_broadcast(A, B)

Contract the last axis of `A` (a single vector or matrix) with the first axis of `B` (any
rank), leaving every other axis free: `(q,)` or `(p, q)` against `(q, ...)`. This is
`numpy.tensordot(A, B, axes=1)`, not `A * B` — the two disagree for rank-3+ `B`.
"""
function _dot_broadcast(A::AbstractArray, B::AbstractArray)
	ndims(A) <= 2 || throw(ArgumentError(
		"Tatva.Linalg._dot_broadcast takes a single vector or matrix as A, got shape $(size(A)). " *
		"$_SINGLE_MATRIX: this builds the full outer product before reducing.",
	))
	q = size(B, 1)
	rest = Base.tail(size(B))
	Bm = reshape(B, q, prod(rest; init = 1))
	if ndims(A) == 1
		r = vec(transpose(A) * Bm)
		return isempty(rest) ? r[1] : reshape(r, rest)
	end
	r = A * Bm
	return isempty(rest) ? vec(r) : reshape(r, (size(A, 1), rest...))
end

# "...ir,jr->...ij": contract the last axes of A and B, B's first axis stays free
function _hessian_transform(A::AbstractArray, B::AbstractMatrix)
	lead = Base.front(size(A))
	Am = reshape(A, prod(lead; init = 1), size(A, ndims(A)))
	return reshape(Am * transpose(B), (lead..., size(B, 1)))
end

#--------------------------------------------------------------------------------
# contract: a fixed table of named contractions
#--------------------------------------------------------------------------------

const _MATMUL = "ab,bc->ac"            # (p, q) × (q, r): element Jacobian and products
const _VECMAT = "a,ab->b"              # (q,) × (q, r): the Line2/Line3 tangent vector
const _ROW_CONTRACT = "ab,b...->a..."  # (p, q) × (q, ...): dNdX against nodal values
const _VEC_CONTRACT = "a,a...->..."    # (q,) × (q, ...): N against nodal values
const _HESSIAN_TRANSFORM = "...ab,cb->...ac"  # (..., i, r) × (j, r) -> (..., i, j)

const _known_implementations = Dict{String, Function}(
	_MATMUL => (A, B) -> A * B,
	_VECMAT => (A, B) -> vec(transpose(A) * B),
	_ROW_CONTRACT => _dot_broadcast,
	_VEC_CONTRACT => _dot_broadcast,
	_HESSIAN_TRANSFORM => _hessian_transform,
)

# (ndims(A), ndims(B)); `nothing` means any rank containing the explicitly named axes
const _known_ranks = Dict{String, Tuple{Union{Nothing, Int}, Union{Nothing, Int}}}(
	_MATMUL => (2, 2),
	_VECMAT => (1, 2),
	_ROW_CONTRACT => (2, nothing),
	_VEC_CONTRACT => (1, nothing),
	_HESSIAN_TRANSFORM => (nothing, 2),
)

# contracted axes as (axis of A, axis of B); :last / :first
const _known_contraction_axes = Dict{String, Tuple{Symbol, Symbol}}(
	_MATMUL => (:last, :first),
	_VECMAT => (:last, :first),
	_ROW_CONTRACT => (:last, :first),
	_VEC_CONTRACT => (:last, :first),
	_HESSIAN_TRANSFORM => (:last, :last),
)

"""Split a spec term into index letters, with `...` as a single token."""
function _tokenize(term::AbstractString)
	tokens = String[]
	chars = collect(term)
	i = 1
	while i <= length(chars)
		if i + 2 <= length(chars) && chars[i] == '.' && chars[i+1] == '.' && chars[i+2] == '.'
			push!(tokens, "...")
			i += 3
		else
			push!(tokens, string(chars[i]))
			i += 1
		end
	end
	return tokens
end

"""
Rename index letters in order of first appearance (`"in,nj->ij"` → `"ab,bc->ac"`), so the
table holds one entry per contraction rather than one per spelling.
"""
function _canonicalise(spec::AbstractString)
	cleaned = replace(spec, " " => "")
	if count("->", cleaned) != 1 || count(',', cleaned) != 1
		throw(ArgumentError(
			"Tatva.Linalg.contract could not parse the spec $(repr(spec)). Expected exactly one ',' " *
			"and one '->', as in 'in,nj->ij'.",
		))
	end
	lhs, out = split(cleaned, "->")
	a_term, b_term = split(lhs, ",")
	renaming = Dict{String, Char}()
	function rename(term)
		letters = String[]
		for token in _tokenize(term)
			if token == "..."
				push!(letters, token)
				continue
			end
			if !(length(token) == 1 && isletter(token[1]))
				throw(ArgumentError(
					"Tatva.Linalg.contract could not parse the spec $(repr(spec)): $(repr(token)) is not " *
					"an index letter.",
				))
			end
			if !haskey(renaming, token)
				renaming[token] = Char('a' + length(renaming))
			end
			push!(letters, string(renaming[token]))
		end
		return join(letters)
	end
	return "$(rename(a_term)),$(rename(b_term))->$(rename(out))"
end

"""
	contract(spec, A, B)

Contract two element-sized tensors, named by an einsum-style spec:

	J    = contract("in,nj->ij", dNdr, coords)          # (d, n) × (n, d)
	dNdX = contract("ij,jn->in", inv(J), dNdr)           # (d, d) × (d, n)
	grad = contract("in,n...->i...", dNdX, nodal_values)  # over the node axis
	val  = contract("n,n...->...", N, nodal_values)      # over the node axis

Index letters are free (`"in,nj->ij"` and `"pq,qr->pr"` are the same entry). An
unrecognised contraction is refused rather than given a general fallback — use
[`einsum`](@ref) for that.
"""
function contract(spec::AbstractString, A::AbstractArray, B::AbstractArray)
	canonical = _canonicalise(spec)
	implementation = get(_known_implementations, canonical, nothing)
	if implementation === nothing
		supported = join(repr.(sort!(collect(keys(_known_implementations)))), ", ")
		throw(ArgumentError(
			"Tatva.Linalg.contract has no implementation for $(repr(spec)) (canonically $(repr(canonical))). " *
			"Supported contractions are $supported. Use Tatva.Linalg.einsum for a general contraction.",
		))
	end

	expected_a, expected_b = _known_ranks[canonical]
	a_term, b_term = split(split(canonical, "->")[1], ",")
	min_a = count(t -> t != "...", _tokenize(a_term))
	min_b = count(t -> t != "...", _tokenize(b_term))
	a_ok = expected_a === nothing ? ndims(A) >= min_a : ndims(A) == expected_a
	b_ok = expected_b === nothing ? ndims(B) >= min_b : ndims(B) == expected_b
	if !(a_ok && b_ok)
		wa = expected_a === nothing ? ">=$min_a" : string(expected_a)
		wb = expected_b === nothing ? ">=$min_b" : string(expected_b)
		throw(ArgumentError(
			"Tatva.Linalg.contract($(repr(spec)), ...) wants operand ranks ($wa, $wb), got " *
			"($(ndims(A)), $(ndims(B))) from shapes $(size(A)) and $(size(B)). These take a single " *
			"tensor each -- batch with map/broadcasting.",
		))
	end

	a_axis, b_axis = _known_contraction_axes[canonical]
	na = size(A, ndims(A))
	nb = b_axis === :first ? size(B, 1) : size(B, ndims(B))
	if na != nb
		throw(DimensionMismatch(
			"Tatva.Linalg.contract($(repr(spec)), ...) contracts A's last axis with B's $(b_axis), " *
			"but they are $na and $nb from shapes $(size(A)) and $(size(B)).",
		))
	end
	return implementation(A, B)
end

#--------------------------------------------------------------------------------
# einsum: drop-in wrapper with a general fallback
#--------------------------------------------------------------------------------

"""
	einsum(spec, operands...)
	einsum(A, sublist_A, B, sublist_B, ..., [sublist_out])

Drop-in einsum. A two-operand spec naming one of the [`contract`](@ref) table entries takes
that path; everything else (more operands, implicit output, unknown contractions, the
interleaved calling convention) is evaluated by a general loop-based einsum. The general
path does not support `...`.
"""
function einsum(spec::AbstractString, operands::AbstractArray...)
	canonical = try
		_canonicalise(spec)
	catch err
		err isa ArgumentError || rethrow()
		nothing
	end
	if canonical !== nothing && haskey(_known_implementations, canonical) && length(operands) == 2
		return contract(spec, operands...)
	end
	return _einsum_general(spec, operands...)
end

function einsum(args...)
	# interleaved convention: (A, [0, 1], B, [1, 2], [0, 2])
	isodd(length(args)) || length(args) >= 2 || throw(ArgumentError("einsum: invalid interleaved arguments"))
	letter(i) = Char('a' + Int(i))
	nops = length(args) ÷ 2
	ops = [args[2k-1] for k in 1:nops]
	terms = [join(letter.(args[2k])) for k in 1:nops]
	spec = join(terms, ",")
	if isodd(length(args))
		spec *= "->" * join(letter.(args[end]))
	end
	return _einsum_general(spec, ops...)
end

function _einsum_general(spec::AbstractString, operands::AbstractArray...)
	cleaned = replace(spec, " " => "")
	occursin("...", cleaned) && throw(ArgumentError("Tatva.Linalg.einsum: '...' is only supported on the contract path"))
	if occursin("->", cleaned)
		lhs, out = split(cleaned, "->")
	else
		lhs = cleaned
		counts = Dict{Char, Int}()
		for c in lhs
			c == ',' && continue
			counts[c] = get(counts, c, 0) + 1
		end
		out = join(sort!([c for (c, n) in counts if n == 1]))
	end
	terms = split(lhs, ",")
	length(terms) == length(operands) || throw(ArgumentError(
		"Tatva.Linalg.einsum: spec $(repr(spec)) names $(length(terms)) operands, got $(length(operands))",
	))
	dims = Dict{Char, Int}()
	for (term, op) in zip(terms, operands)
		length(term) == ndims(op) || throw(DimensionMismatch(
			"Tatva.Linalg.einsum: term $(repr(term)) does not match operand of shape $(size(op))",
		))
		for (k, c) in enumerate(term)
			n = size(op, k)
			if haskey(dims, c) && dims[c] != n
				throw(DimensionMismatch("Tatva.Linalg.einsum: index '$c' has inconsistent sizes $(dims[c]) and $n"))
			end
			dims[c] = n
		end
	end
	out_letters = collect(out)
	summed = sort!([c for c in keys(dims) if !(c in out_letters)])
	all_letters = vcat(out_letters, summed)
	pos = Dict(c => i for (i, c) in enumerate(all_letters))
	T = promote_type(map(eltype, operands)...)
	out_shape = Tuple(dims[c] for c in out_letters)
	result = zeros(T, out_shape)
	ranges = Tuple(1:dims[c] for c in all_letters)
	term_pos = [Tuple(pos[c] for c in term) for term in terms]
	for I in CartesianIndices(ranges)
		v = one(T)
		for (op, tp) in zip(operands, term_pos)
			v *= op[ntuple(k -> I[tp[k]], length(tp))...]
		end
		result[ntuple(k -> I[k], length(out_letters))...] += v
	end
	return isempty(out_shape) ? result[] : result
end

end # module Linalg
