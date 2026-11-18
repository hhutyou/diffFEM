#--------------------------------------------------------------------------------
# Automatic differentiation front end
#--------------------------------------------------------------------------------
#
# tatva builds every operator by differentiating an energy (jax.grad / jacrev / jacfwd).
# Here forward mode is ForwardDiff's dual numbers, driven by a small chunked driver that is
# safe to nest: results are allocated with the type of the *extracted* partials, so an
# inner derivative evaluated on dual numbers of an outer differentiation returns those
# outer duals (ForwardDiff orders tags by creation, the innermost differentiation being the
# outermost dual). Reverse mode is available through the Enzyme package extension
# (`backend = AutoEnzyme()`).

"""
	ArgDerivative(f, argnum, backend)

Callable returned by [`jacrev`](@ref): the derivative of `f` with respect to its
`argnum`-th positional argument.
"""
struct ArgDerivative{F, I, B <: AbstractADType}
	f::F
	backend::B
end
ArgDerivative(f, argnum::Integer, backend::AbstractADType) = ArgDerivative{typeof(f), Int(argnum), typeof(backend)}(f, backend)

"""
	jacrev(f; argnums=1, backend=AutoForwardDiff())

Return a function computing the derivative of `f` with respect to its `argnums`-th
argument, at the given arguments (JAX's `jax.jacrev`). For a scalar-valued `f` this is the
gradient, with the shape of the argument; for an array-valued `f` the Jacobian, of shape
`(size(y)..., size(x)...)`.

With the default `AutoForwardDiff()` backend the gradient is computed in chunks of
forward-mode passes; load Enzyme and pass `backend = AutoEnzyme()` for reverse mode.
"""
jacrev(f; argnums::Integer = 1, backend::AbstractADType = AutoForwardDiff()) = ArgDerivative(f, argnums, backend)

function (d::ArgDerivative{F, I})(args...; kwargs...) where {F, I}
	x = args[I]
	h = _ArgClosure{F, I, typeof(args), typeof(values(kwargs))}(d.f, args, values(kwargs))
	return derivative(d.backend, h, x)
end

# `ξ -> f(args with args[I] = ξ...)`, as a concrete callable so that ForwardDiff tags are
# specific to the captured argument types.
struct _ArgClosure{F, I, A, K}
	f::F
	args::A
	kwargs::K
end
@inline (c::_ArgClosure{F, I})(ξ) where {F, I} = c.f(_setarg(c.args, Val(I), ξ)...; c.kwargs...)
@inline _setarg(args::Tuple, ::Val{I}, x) where {I} = ntuple(k -> k == I ? x : args[k], Val(length(args)))

"""
	derivative(backend, f, x)

Derivative of the single-argument function `f` at `x`: the gradient (shape of `x`) for a
scalar `f`, the Jacobian `(size(y)..., size(x)...)` for an array-valued `f`.
"""
function derivative end

derivative(backend::AbstractADType, f, x) = throw(ArgumentError(
	"AD backend $(backend) is not available. Tatva supports AutoForwardDiff() out of the box; " *
	"AutoEnzyme() requires `using Enzyme`."))

function derivative(backend::AutoForwardDiff{CS}, f::F, x::Number) where {CS, F}
	T = typeof(x)
	Tg = typeof(ForwardDiff.Tag(f, T))
	y = f(ForwardDiff.Dual{Tg}(x, one(x)))
	return _extract_partial(Tg, y, 1)
end

function derivative(backend::AutoForwardDiff{CS}, f::F, x::AbstractArray{T}) where {CS, F, T}
	n = length(x)
	n == 0 && throw(ArgumentError("cannot differentiate with respect to an empty array"))
	C = CS === nothing ? ForwardDiff.pickchunksize(n) : min(CS, n)
	Tg = typeof(ForwardDiff.Tag(f, T))
	return _chunked_derivative(f, x, Tg, Val(C))
end

@inline _extract_partial(::Type{Tg}, y::Number, k) where {Tg} = ForwardDiff.partials(Tg, y, k)
@inline _extract_partial(::Type{Tg}, y::AbstractArray, k) where {Tg} = map(yi -> ForwardDiff.partials(Tg, yi, k), y)

function _chunked_derivative(f::F, x::AbstractArray{T}, ::Type{Tg}, ::Val{C}) where {F, T, Tg, C}
	n = length(x)
	xd = similar(x, ForwardDiff.Dual{Tg, T, C})
	_seed_chunk!(xd, x, 1, Val(C))
	y = f(xd)
	result = _alloc_derivative(Tg, y, x)
	_store_chunk!(result, Tg, y, 1, min(C, n))
	return _chunked_derivative_rest!(result, f, xd, x, Tg, Val(C))
end

function _chunked_derivative_rest!(result, f::F, xd, x, ::Type{Tg}, ::Val{C}) where {F, Tg, C}
	n = length(x)
	for c0 ∈ (1+C):C:n
		_seed_chunk!(xd, x, c0, Val(C))
		y = f(xd)
		_store_chunk!(result, Tg, y, c0, min(C, n - c0 + 1))
	end
	return result
end

# seed x[c0 : c0+C-1] with the unit perturbations
function _seed_chunk!(xd::AbstractArray{ForwardDiff.Dual{Tg, T, C}}, x, c0::Int, ::Val{C}) where {Tg, T, C}
	@inbounds for (i, xi) ∈ enumerate(x)
		k = i - c0 + 1
		xd[i] = ForwardDiff.Dual{Tg}(xi, ForwardDiff.Partials{C, T}(ntuple(j -> j == k ? one(T) : zero(T), Val(C))))
	end
	return xd
end

_alloc_derivative(::Type{Tg}, y::Number, x) where {Tg} = similar(x, typeof(ForwardDiff.partials(Tg, y, 1)))
function _alloc_derivative(::Type{Tg}, y::AbstractArray, x) where {Tg}
	P = typeof(ForwardDiff.partials(Tg, first(y), 1))
	return Array{P}(undef, size(y)..., size(x)...)
end

function _store_chunk!(result::AbstractArray, ::Type{Tg}, y::Number, c0::Int, w::Int) where {Tg}
	@inbounds for k ∈ 1:w
		result[c0+k-1] = ForwardDiff.partials(Tg, y, k)
	end
	return result
end
function _store_chunk!(result::AbstractArray, ::Type{Tg}, y::AbstractArray, c0::Int, w::Int) where {Tg}
	m = length(y)
	R = reshape(result, m, :)
	@inbounds for k ∈ 1:w, i ∈ 1:m
		R[i, c0+k-1] = ForwardDiff.partials(Tg, y[i], k)
	end
	return result
end
