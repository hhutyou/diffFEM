#--------------------------------------------------------------------------------
# Lifter: map between reduced (free) and full dof vectors
#--------------------------------------------------------------------------------

"""Error raised for an inconsistent lifter, e.g. a missing runtime value."""
struct LifterError <: Exception
	msg::String
end
Base.showerror(io::IO, e::LifterError) = print(io, "LifterError: ", e.msg)

"""
	RuntimeValue(key=uuid4(), default=nothing)

Placeholder for a constraint value provided at runtime (e.g. a load-step dependent
Dirichlet value). Set it with `with_values(lifter, key => value)`; without a `default`
lifting raises a [`LifterError`](@ref) until it is set.
"""
struct RuntimeValue{K}
	key::K
	default::Any
end
RuntimeValue() = RuntimeValue(uuid4(), nothing)
RuntimeValue(key) = RuntimeValue(key, nothing)

function get_value(rv::RuntimeValue, runtime_values::AbstractDict)
	haskey(runtime_values, rv.key) || throw(LifterError("Runtime value for (key=$(repr(rv.key))) not set on lifter"))
	return runtime_values[rv.key]
end

"""
	Constraint

Supertype of conditions applied during lifting. A constraint has `dofs` (the constrained
dofs) and implements `apply_lift!(c, u_full, runtime_values)` and
`apply_transpose!(c, r_full)`; it may also implement `augment_sparsity(c, pattern)`.
Constraints compare by identity (a unique id assigned at construction).
"""
abstract type Constraint end

Base.:(==)(a::Constraint, b::Constraint) = typeof(a) == typeof(b) && a.id == b.id
Base.hash(c::Constraint, h::UInt) = hash(c.id, hash(nameof(typeof(c)), h))

_dof_vector(dofs::Integer) = [Int(dofs)]
_dof_vector(dofs) = Vector{Int}(vec(collect(dofs)))

"""
	Fixed(dofs, values=0.0)

Dirichlet condition: set the full vector at `dofs` to `values` (a number, an array with one
value per dof, or a [`RuntimeValue`](@ref)).
"""
struct Fixed{V} <: Constraint
	dofs::Vector{Int}
	values::V
	id::UUID
end
Fixed(dofs, values = 0.0) = Fixed(_dof_vector(dofs), values, uuid4())

"""
	Periodic(dofs, master_dofs)
	Periodic(; dofs, master_dofs)

Periodicity: the full vector at `dofs` is copied from `master_dofs` (element-wise, all
masters read before any dof is written). A dof may appear only once in `dofs`.
"""
struct Periodic <: Constraint
	dofs::Vector{Int}
	master_dofs::Vector{Int}
	id::UUID
end
function Periodic(dofs, master_dofs)
	d, m = _dof_vector(dofs), _dof_vector(master_dofs)
	length(d) == length(m) || throw(DimensionMismatch("Periodic: $(length(d)) dofs but $(length(m)) master dofs"))
	allunique(d) || throw(ArgumentError("Periodic: a dof appears twice in `dofs` (it would have two masters)"))
	return Periodic(d, m, uuid4())
end
Periodic(; dofs, master_dofs) = Periodic(dofs, master_dofs)

"""
	PeriodicMPI(dofs, master_dofs, layout; comm)

Periodicity across MPI ranks, specified with natural global dof indices (e.g. from
`global_indices(Compound).u[...]`). Slaves present on this rank are constrained locally;
masters not present locally become extra ghost dofs when the lifter adapts the layout with
[`adapt_layout`](@ref).
"""
struct PeriodicMPI{C} <: Constraint
	dofs::Vector{Int}
	master_dofs::Vector{Int}
	slave_natural::Vector{Int}
	master_natural::Vector{Int}
	extra_ghost_dofs::Vector{Int}
	comm::C
	id::UUID
end

#--------------------------------------------------------------------------------
# constraint actions
#--------------------------------------------------------------------------------

_resolve(v::RuntimeValue, runtime_values) = get_value(v, runtime_values)
_resolve(v, runtime_values) = v

"""Collect the `RuntimeValue`s referenced by a constraint."""
_runtime_specs(c::Constraint) = _collect_runtime(c, RuntimeValue[])
function _collect_runtime(x, acc)
	if x isa RuntimeValue
		push!(acc, x)
	elseif x isa Union{Tuple, AbstractArray{Any}, AbstractArray{<:RuntimeValue}}
		foreach(y -> _collect_runtime(y, acc), x)
	elseif x isa AbstractDict
		foreach(y -> _collect_runtime(y, acc), values(x))
	elseif x isa Constraint
		foreach(name -> _collect_runtime(getfield(x, name), acc), fieldnames(typeof(x)))
	end
	return acc
end

apply_lift!(::Constraint, u, runtime_values) = u
apply_transpose!(::Constraint, r) = r
augment_sparsity(::Constraint, pattern::SparseMatrixCSC) = pattern

function apply_lift!(c::Fixed, u, runtime_values)
	vals = _resolve(c.values, runtime_values)
	if vals isa Number
		u[c.dofs] .= vals
	else
		length(vals) == length(c.dofs) || throw(DimensionMismatch("Fixed: $(length(c.dofs)) dofs but $(length(vals)) values"))
		u[c.dofs] .= vec(vals)
	end
	return u
end
apply_transpose!(c::Fixed, r) = (r[c.dofs] .= zero(eltype(r)); r)

const _AnyPeriodic = Union{Periodic, PeriodicMPI}

apply_lift!(c::_AnyPeriodic, u, runtime_values) = (u[c.dofs] = u[c.master_dofs]; u)

# transpose of `u[dofs] = u[master_dofs]`: the slave entries are gathered and cleared before
# they are added to the masters, as a master may itself be a slave of the same constraint
function apply_transpose!(c::_AnyPeriodic, r)
	slave_values = r[c.dofs]
	r[c.dofs] .= zero(eltype(r))
	for (m, v) ∈ zip(c.master_dofs, slave_values)
		r[m] += v
	end
	return r
end

function augment_sparsity(c::_AnyPeriodic, pattern::SparseMatrixCSC)
	n = size(pattern, 1)
	# u_full = M u with M = I + E (Eᵢⱼ = 1 for slave i of master j): the reduced operator is
	# Mᵀ S M, so masters inherit the couplings of their slaves.
	M = sparse(vcat(1:n, c.dofs), vcat(1:n, c.master_dofs), ones(Int, n + length(c.dofs)), n, n)
	S = SparseMatrixCSC{Int, Int}(pattern .!= 0)
	return _binarize(transpose(M) * (S * M))
end

_binarize(A::AbstractSparseMatrix) = (B = sparse(A); dropzeros!(B); SparseMatrixCSC{Bool, Int}(size(B)..., B.colptr, B.rowval, fill(true, nnz(B))))

#--------------------------------------------------------------------------------
# Lifter
#--------------------------------------------------------------------------------

"""
	Lifter(size, constraints...)

Map between the reduced vector of free dofs and the full vector of `size` dofs. Lifting sets
the free dofs from the reduced vector and then applies the constraints in order.

	lifter = Lifter(6, Fixed([1, 6], 0.0), Periodic([3], [2]))
	u_full = lift_from_zeros(lifter, [10.0, 20.0, 30.0])  # [0, 10, 10, 20, 30, 0]
	reduce(lifter, u_full)                               # [10, 20, 30]
"""
struct Lifter{C <: Tuple}
	size::Int
	constraints::C
	free_dofs::Vector{Int}
	constrained_dofs::Vector{Int}
	size_reduced::Int
	runtime_keys::Vector{Any}
	runtime_values::Dict{Any, Any}
	nb_extra_ghost_dofs::Union{Nothing, Int}
end

function Lifter(size::Integer, constraints::Constraint...; nb_extra_ghost_dofs::Union{Nothing, Integer} = nothing)
	specs = reduce(vcat, map(_runtime_specs, collect(constraints)); init = RuntimeValue[])
	keys_ = Any[s.key for s in specs]
	runtime_values = Dict{Any, Any}(s.key => s.default for s in specs if s.default !== nothing)
	free, constrained = _compute_free_dofs(Int(size), constraints)
	return Lifter(Int(size), Tuple(constraints), free, constrained, length(free), keys_, runtime_values,
		nb_extra_ghost_dofs === nothing ? nothing : Int(nb_extra_ghost_dofs))
end

function _compute_free_dofs(size::Int, constraints)
	isempty(constraints) && return collect(1:size), Int[]
	constrained = sort!(unique(reduce(vcat, [c.dofs for c in constraints])))
	isempty(constrained) || (1 <= first(constrained) && last(constrained) <= size) ||
		throw(LifterError("constrained dofs must lie in 1:$size"))
	mask = trues(size)
	mask[constrained] .= false
	return findall(mask), constrained
end

Base.show(io::IO, l::Lifter) = print(io, "Lifter(size=", l.size, ", size_reduced=", l.size_reduced, ", ",
	length(l.constraints), " constraint", length(l.constraints) == 1 ? "" : "s", ")")

function Base.:(==)(a::Lifter, b::Lifter)
	a.size == b.size && a.constraints == b.constraints && a.nb_extra_ghost_dofs == b.nb_extra_ghost_dofs &&
		_runtime_values_equal(a.runtime_values, b.runtime_values)
end
Base.hash(l::Lifter, h::UInt) = hash((l.size, l.constraints, l.nb_extra_ghost_dofs), h)

function _runtime_values_equal(a::AbstractDict, b::AbstractDict)
	keys(a) == keys(b) || return false
	return all(k -> (a[k] === b[k]) || (size(a[k]) == size(b[k]) && all(a[k] .== b[k])), keys(a))
end

"""Size of the full vector returned by `lift` (excludes extra MPI ghost dofs)."""
local_size(l::Lifter) = l.size - something(l.nb_extra_ghost_dofs, 0)

"""
	add(lifter, constraint)

A new lifter with `constraint` appended (runtime values already set are kept).
"""
add(l::Lifter, c::Constraint) = with_values(Lifter(l.size, l.constraints..., c), l.runtime_values)

# element type needed to hold the lifted vector
function _constraint_eltype(l::Lifter)
	T = Union{}
	for c ∈ l.constraints
		if c isa Fixed
			T = promote_type(T, eltype(_resolve(c.values, l.runtime_values)))
		end
	end
	return T
end

"""
	lift(lifter, u_reduced, u_full)

Full vector with the free dofs set to `u_reduced` and the constraints applied, starting
from `u_full` (typically the previous solution).
"""
function lift(l::Lifter, u_reduced::AbstractVector, u_full::AbstractVector)
	length(u_reduced) == l.size_reduced ||
		throw(DimensionMismatch("lift: reduced vector has length $(length(u_reduced)), expected $(l.size_reduced)"))
	length(u_full) == l.size || throw(DimensionMismatch("lift: full vector has length $(length(u_full)), expected $(l.size)"))
	T = promote_type(eltype(u_reduced), eltype(u_full), _constraint_eltype(l))
	return _lift!(Vector{T}(undef, l.size), l, u_reduced, u_full)
end

function _lift!(u::AbstractVector, l::Lifter, u_reduced, u_full)
	copyto!(u, u_full)
	u[l.free_dofs] .= u_reduced
	for c ∈ l.constraints
		apply_lift!(c, u, l.runtime_values)
	end
	return l.nb_extra_ghost_dofs === nothing ? u : u[1:local_size(l)]
end

"""
	lift_from_zeros(lifter, u_reduced)

Lift a reduced vector into a full vector starting from zeros.
"""
lift_from_zeros(l::Lifter, u_reduced::AbstractVector) = lift(l, u_reduced, zeros(eltype(u_reduced), l.size))

# `P v`, the linear part of the lift: the lift of zero (the fixed values) cancels
_lift_linear(l::Lifter, v::AbstractVector) = lift_from_zeros(l, v) - lift_from_zeros(l, zeros(eltype(v), l.size_reduced))

"""
	reduce(lifter, u_full)

Reduced vector: the free dofs of `u_full`.
"""
Base.reduce(l::Lifter, u_full::AbstractVector) = u_full[l.free_dofs]

"""
	reduce_adjoint(lifter, r_full)

Reduce a dual vector (residual, gradient): apply the transpose of every constraint in
reverse order (e.g. periodic slave residuals are added to their masters) and keep the free
dofs. This is the adjoint of `lift`, so `reduce_adjoint(l, ∇E(lift(l, u)))` is the gradient
of the reduced energy.
"""
function reduce_adjoint(l::Lifter, r_full::AbstractVector)
	r = copyto!(similar(r_full, eltype(r_full), length(r_full)), r_full)
	for c ∈ reverse(l.constraints)
		apply_transpose!(c, r)
	end
	return r[l.free_dofs]
end

"""
	with_values(lifter, updates)
	with_values(lifter, key => value, ...)

A new lifter with runtime values updated. Unknown keys raise a [`LifterError`](@ref).
"""
function with_values(l::Lifter, updates::AbstractDict; strict::Bool = true)
	if strict
		for key ∈ keys(updates)
			key in l.runtime_keys || throw(LifterError("There is no runtime value with key=$(repr(key)) in the lifter's constraints"))
		end
	end
	values_ = merge(l.runtime_values, Dict{Any, Any}(updates))
	return Lifter(l.size, l.constraints, l.free_dofs, l.constrained_dofs, l.size_reduced, l.runtime_keys, values_, l.nb_extra_ghost_dofs)
end
with_values(l::Lifter, updates::Pair...) = with_values(l, Dict{Any, Any}(updates...))

# `lifter.at[key].set(value)`, as in tatva
struct RuntimeValueIndexer
	lifter::Lifter
end
struct RuntimeValueSetter
	lifter::Lifter
	key::Any
end
Base.getindex(idx::RuntimeValueIndexer, key) = RuntimeValueSetter(idx.lifter, key)
(idx::RuntimeValueIndexer)(key) = RuntimeValueSetter(idx.lifter, key)
set(s::RuntimeValueSetter, value) = with_values(s.lifter, s.key => value)
Base.getproperty(s::RuntimeValueSetter, name::Symbol) = name === :set ? (value -> set(s, value)) : getfield(s, name)
Base.getproperty(l::Lifter, name::Symbol) = name === :at ? RuntimeValueIndexer(l) : getfield(l, name)
Base.propertynames(l::Lifter) = (fieldnames(Lifter)..., :at)

#--------------------------------------------------------------------------------
# sparsity patterns
#--------------------------------------------------------------------------------

"""
	augment_sparsity(lifter, pattern)

Augment a full sparsity pattern with the couplings introduced by the constraints (only
periodic constraints add entries): the result contains the pattern and that of `Pᵀ S P`,
`P` the linear part of the lift. A pattern smaller than `lifter.size` is padded first.
"""
function augment_sparsity(l::Lifter, pattern::SparseMatrixCSC)
	S = pattern
	if size(S, 1) < l.size
		I, J, _ = findnz(S)
		S = sparse(I, J, fill(true, length(I)), l.size, l.size)
	end
	# the lift applies the constraints in order, P = M_K ⋯ M_1, so Pᵀ S P pulls the pattern
	# back through the last constraint first (a slave can be the master of a later constraint)
	for c ∈ reverse(l.constraints)
		S = augment_sparsity(c, S)
	end
	return _binarize(S)
end

"""
	reduce_sparsity(lifter, pattern)

Restrict a full pattern to the free dofs.
"""
reduce_sparsity(l::Lifter, pattern::SparseMatrixCSC) = _binarize(pattern[l.free_dofs, l.free_dofs])

"""
	adapt_sparsity(lifter, pattern)

`reduce_sparsity(lifter, augment_sparsity(lifter, pattern))`: the pattern of the reduced
operator.
"""
adapt_sparsity(l::Lifter, pattern::SparseMatrixCSC) = reduce_sparsity(l, augment_sparsity(l, pattern))

#--------------------------------------------------------------------------------
# lifted functions
#--------------------------------------------------------------------------------

"""
	lifted(fn; argnums=1, output=nothing) -> (lifter, args...; kwargs...) -> ...

Turn a function of full vectors into one of reduced vectors: the arguments at positions
`argnums` are lifted with `lift_from_zeros` before calling `fn`. With `output = :primal` the
result is reduced with `reduce`, with `output = :dual` with [`reduce_adjoint`](@ref)
(residuals, gradients).
"""
function lifted(fn; argnums = 1, output::Union{Nothing, Symbol, AbstractString} = nothing)
	positions = Tuple(argnums)
	out = output === nothing ? nothing : Symbol(output)
	out in (nothing, :primal, :dual) || throw(LifterError("Invalid value for output: $(repr(output))"))
	return function lifted_fn(lifter::Lifter, args...; kwargs...)
		lifted_args = ntuple(length(args)) do i
			i in positions || return args[i]
			arg = args[i]
			arg isa AbstractVector || throw(LifterError("Argument $i is not a vector and cannot be lifted by the lifter"))
			length(arg) == lifter.size_reduced ||
				throw(LifterError("Argument $i has shape $(size(arg)) but expected ($(lifter.size_reduced),) for lifting"))
			lift_from_zeros(lifter, arg)
		end
		result = fn(lifted_args...; kwargs...)
		out === nothing && return result
		result isa AbstractVector || throw(LifterError("Output is not a vector and cannot be reduced by the lifter"))
		return out === :primal ? reduce(lifter, result) : reduce_adjoint(lifter, result)
	end
end
