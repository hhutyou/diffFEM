#--------------------------------------------------------------------------------
# Automatic sparsity detection from an energy / virtual-work form
#--------------------------------------------------------------------------------
#
# tatva interprets the jaxpr of the energy with a handler per JAX primitive. In Julia the
# same idea is operator overloading: SparseConnectivityTracer.jl runs the (generic) energy
# on tracer numbers that carry first-order dependency sets and record second-order
# couplings. Index routing (gather/scatter), loops (lax.map/scan, including carries),
# function barriers (jit/remat) and custom derivative rules need no special handling: the
# code simply executes on tracers. What does need help:
#
# - data-dependent control flow: the global tracer cannot branch on a tracer, so use
#   `branch`/`switch` (both branches are traced and unioned, like tatva's `cond` handler) or
#   `ifelse` instead of `if`;
# - non-generic code (C/Fortran libraries, LAPACK on tracers such as `eigvals`, `sort`):
#   wrap it in `opaque(f, args...)`, which conservatively couples all its inputs (tatva's
#   handling of callbacks / custom_vjp / ffi_call leaves). Map it per quadrature point to
#   keep the pattern block-diagonal (tatva's `register_elementwise_ffi`).

import SparseConnectivityTracer as SCT
using SparseConnectivityTracer: TracerSparsityDetector, TracerLocalSparsityDetector

const _AnyTracer = Union{SCT.AbstractTracer, SCT.Dual}

# SparseConnectivityTracer 1.2 defines promote_rule(::Type{<:Real}, ::Type{<:AbstractTracer}),
# which is ambiguous with Base's rules for Bool and irrationals, so `I + ∇u` (UniformScaling{Bool})
# or `π * x` fail on tracers. Add the disambiguating methods only while the ambiguity exists.
_promote_ambiguous(S, T) =
	try
		promote_rule(S, T)
		false
	catch err
		err isa MethodError || rethrow()
		true
	end

let H = SCT.hessian_eltype(zeros(1), TracerSparsityDetector()),
	D = SCT.hessian_eltype(zeros(1), TracerLocalSparsityDetector())

	if _promote_ambiguous(Bool, H)
		@eval Base.promote_rule(::Type{Bool}, ::Type{T}) where {T <: SCT.AbstractTracer} = T
	end
	if _promote_ambiguous(typeof(π), H)
		@eval Base.promote_rule(::Type{S}, ::Type{T}) where {S <: AbstractIrrational, T <: SCT.AbstractTracer} = T
	end
	if _promote_ambiguous(Bool, D)
		@eval Base.promote_rule(::Type{Bool}, ::Type{D}) where {P, T, D <: SCT.Dual{P, T}} = SCT.Dual{promote_type(Bool, P), T}
	end
	if _promote_ambiguous(typeof(π), D)
		@eval Base.promote_rule(::Type{S}, ::Type{D}) where {S <: AbstractIrrational, P, T, D <: SCT.Dual{P, T}} =
			SCT.Dual{promote_type(S, P), T}
	end
end

_identity_pattern(n::Integer) = sparse(1:n, 1:n, fill(true, n), n, n)

"""
	pattern_from_energy(energy_fn, n_dofs, static_args...; detector=TracerSparsityDetector(), x=nothing)

Sparsity pattern of the Hessian `∂²E/∂u²` of the scalar energy `energy_fn(u, static_args...)`
(`u` of length `n_dofs`), as a symmetric `SparseMatrixCSC{Bool}`. The default global
detector gives a pattern valid for every `u` (no false negatives); a
`TracerLocalSparsityDetector()` gives the (possibly sparser) pattern at the point `x`.
If nothing couples, the identity is returned (as in tatva).
"""
function pattern_from_energy(energy_fn, n_dofs::Integer, static_args...; detector = TracerSparsityDetector(), x = nothing)
	x0 = x === nothing ? zeros(Int(n_dofs)) : x
	H = SCT.hessian_sparsity(u -> energy_fn(u, static_args...), x0, detector)
	return nnz(H) == 0 ? _identity_pattern(Int(n_dofs)) : SparseMatrixCSC{Bool, Int}(H)
end

"""
	pattern_from_virtual_work(vw_fn, n_dofs, trial_arg, test_arg, static_args...; detector=TracerSparsityDetector())

Sparsity pattern of the tangent `K = ∂R/∂u = ∂²G/∂v∂u` of a virtual-work form `G(…, u, …, v, …)`
that is linear in the test function `v`. `trial_arg`/`test_arg` are argument names
(`Symbol`/`String`, looked up in the method signature) or 1-based positions. The remaining
parameters are filled in order from `static_args`; trailing parameters with default values
may be omitted. Rows are test dofs, columns trial dofs (the pattern may be unsymmetric).
"""
function pattern_from_virtual_work(vw_fn, n_dofs::Integer, trial_arg, test_arg, static_args...; detector = TracerSparsityDetector())
	names, ms = _argument_names(vw_fn)
	trial_idx = _arg_position(trial_arg, names, "Trial", vw_fn)
	test_idx = _arg_position(test_arg, names, "Test", vw_fn)
	trial_idx == test_idx && throw(ArgumentError("trial and test argument must differ"))

	plan = Any[]
	k = 0
	for p ∈ eachindex(names)
		if p == trial_idx
			push!(plan, :trial)
		elseif p == test_idx
			push!(plan, :test)
		elseif k < length(static_args)
			k += 1
			push!(plan, Ref(static_args[k]))
		else
			push!(plan, nothing)
		end
	end
	last_needed = findlast(!isnothing, plan)
	for p ∈ 1:last_needed
		plan[p] === nothing && throw(ArgumentError("Missing static argument for parameter '$(names[p])' in $(_fname(vw_fn))"))
	end
	nargs = last_needed
	if !any(m -> m.nargs - 1 == nargs || m.isva, ms)
		throw(ArgumentError("Missing static argument for parameter '$(names[nargs + 1])' in $(_fname(vw_fn))"))
	end
	plan = plan[1:nargs]

	n = Int(n_dofs)
	function w_fn(w)
		u = w[1:n]
		v = w[(n+1):(2n)]
		args = map(s -> s === :trial ? u : s === :test ? v : s[], plan)
		return vw_fn(args...)
	end
	H = SCT.hessian_sparsity(w_fn, zeros(2n), detector)
	return SparseMatrixCSC{Bool, Int}(H[(n+1):(2n), 1:n])
end

_fname(f) = f isa Function ? string(nameof(f)) : string(typeof(f))

function _argument_names(f)
	ms = collect(methods(f))
	isempty(ms) && throw(ArgumentError("$(_fname(f)) has no methods"))
	m = ms[argmax([m.nargs for m in ms])]
	return Base.method_argnames(m)[2:end], ms
end

function _arg_position(arg, names, what, f)
	if arg isa Integer
		1 <= arg <= length(names) || throw(ArgumentError("$what argument position $arg out of range for $(_fname(f))"))
		return Int(arg)
	end
	s = Symbol(arg)
	p = findfirst(==(s), names)
	p === nothing && throw(ArgumentError(
		"$what argument '$s' not found in signature of $(_fname(f)). Available parameters: $(names)"))
	return p
end

#--------------------------------------------------------------------------------
# control flow and opaque functions
#--------------------------------------------------------------------------------

"""
	branch(pred, f_true, f_false, operands...)

`pred ? f_true(operands...) : f_false(operands...)` (JAX's `lax.cond`), usable in energies
that are traced for sparsity: when `pred` is a sparsity tracer both branches are evaluated
and their dependencies unioned.
"""
branch(pred::Bool, f_true, f_false, operands...) = pred ? f_true(operands...) : f_false(operands...)
branch(pred::SCT.AbstractTracer, f_true, f_false, operands...) = _branch_union(f_true(operands...), f_false(operands...))
branch(pred::Number, f_true, f_false, operands...) = branch(!iszero(pred), f_true, f_false, operands...)

"""
	switch(index, branches, operands...)

`branches[index](operands...)` with the (1-based) index clamped to the valid range (JAX's
`lax.switch`); with a tracer index all branches are evaluated and unioned.
"""
switch(index::Integer, branches, operands...) = branches[clamp(Int(index), 1, length(branches))](operands...)
switch(index::SCT.AbstractTracer, branches, operands...) = reduce(_branch_union, map(b -> b(operands...), Tuple(branches)))
switch(index::Real, branches, operands...) = switch(round(Int, index), branches, operands...)

_branch_union(a::Tuple, b::Tuple) = map(_branch_union, a, b)
_branch_union(a, b) = SCT.output_union(a, b)

"""
	opaque(f, args...; out=nothing)

Call `f(args...)`. When the arguments carry sparsity tracers, `f` is not traced: every
output depends on — and every pair of inputs is coupled through — all traced inputs (the
conservative rule tatva applies to callbacks and custom-derivative leaves). Use it for code
that cannot run on tracers (foreign libraries, LAPACK-based `eigvals`, sorting, …). The
output shape is found by calling `f` on placeholder values (zeros, then ones); pass a
prototype output `out` if that fails.
"""
function opaque(f, args...; out = nothing)
	_has_tracer(args) || return f(args...)
	tracers = _collect_tracers!(Any[], args)
	s = sum(tracers)
	t = s * s  # gradient: union of all inputs; Hessian: their full outer product
	proto = out === nothing ? _opaque_prototype(f, args) : out
	return _fill_tracer(proto, t)
end

_has_tracer(x::_AnyTracer) = true
_has_tracer(x::AbstractArray) = _type_has_tracer(eltype(x)) || (!isconcretetype(eltype(x)) && any(_has_tracer, x))
_type_has_tracer(::Type{T}) where {T} = T <: _AnyTracer || (T <: StaticArray && _type_has_tracer(eltype(T)))
_has_tracer(x::Tuple) = any(_has_tracer, x)
_has_tracer(x) = false

_collect_tracers!(acc, x::_AnyTracer) = push!(acc, x)
_collect_tracers!(acc, x::Union{AbstractArray, Tuple}) = (foreach(y -> _collect_tracers!(acc, y), x); acc)
_collect_tracers!(acc, x) = acc

_placeholder(x::SCT.Dual, v) = SCT.primal(x)
_placeholder(x::SCT.AbstractTracer, v) = v
_placeholder(x::AbstractArray, v) = map(y -> _placeholder(y, v), x)
_placeholder(x::Tuple, v) = map(y -> _placeholder(y, v), x)
_placeholder(x, v) = x

function _opaque_prototype(f, args)
	for v ∈ (0.0, 1.0)
		try
			return f(map(a -> _placeholder(a, v), args)...)
		catch err
			err isa InterruptException && rethrow()
		end
	end
	throw(ArgumentError("opaque: could not infer the output shape of $(_fname(f)); pass a prototype with `out=`"))
end

_fill_tracer(proto::Number, t::SCT.AbstractTracer) = t
_fill_tracer(proto::Number, t::SCT.Dual) = SCT.Dual(proto, SCT.tracer(t))
_fill_tracer(proto::AbstractArray, t) = map(p -> _fill_tracer(p, t), proto)
_fill_tracer(proto::Tuple, t) = map(p -> _fill_tracer(p, t), proto)
