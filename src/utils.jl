#--------------------------------------------------------------------------------
# Utilities
#--------------------------------------------------------------------------------

"""
	virtual_work_to_residual(fn; test_arr=nothing, test_shape=nothing, test_size=nothing, backend=AutoForwardDiff())
	virtual_work_to_residual(; test_size=n, ...) do test, args...
		...
	end

Turn a virtual-work function `fn(test, args...)`, linear in the test function (first
argument), into the residual `R(args...) = ∂fn/∂test` (evaluated at `test_arr`, or zeros of
`test_shape`/`test_size`). Exactly one of the three must be given. The result can be fed to
[`jacfwd`](@ref) to obtain the (possibly unsymmetric) tangent.
"""
function virtual_work_to_residual(fn = nothing; test_arr = nothing, test_shape = nothing, test_size = nothing,
	backend::AbstractADType = AutoForwardDiff())
	t = if test_arr !== nothing
		test_arr isa Number ? test_arr : collect(test_arr)
	elseif test_shape !== nothing
		zeros(Tuple(test_shape)...)
	elseif test_size !== nothing
		zeros(Int(test_size))
	else
		throw(ArgumentError("One of 'test_arr', 'test_shape', or 'test_size' must be provided."))
	end
	fn === nothing && return f -> VirtualWorkResidual(f, t, backend)
	return VirtualWorkResidual(fn, t, backend)
end

"""Residual `args -> ∂fn(test, args...)/∂test` built by [`virtual_work_to_residual`](@ref)."""
struct VirtualWorkResidual{F, T, B <: AbstractADType}
	fn::F
	test_arr::T
	backend::B
end

function (r::VirtualWorkResidual{F})(args...; kwargs...) where {F}
	h = _TestClosure{F, typeof(args), typeof(values(kwargs))}(r.fn, args, values(kwargs))
	return derivative(r.backend, h, r.test_arr)
end

struct _TestClosure{F, A, K}
	fn::F
	args::A
	kwargs::K
end
@inline (c::_TestClosure)(test) = c.fn(test, c.args...; c.kwargs...)

#--------------------------------------------------------------------------------
# L2 projection
#--------------------------------------------------------------------------------

"""
	make_project_function(n_nodes; colored_matrix=nothing, elements=nothing, lifter=nothing) -> (op, field) -> nodal

Factory for the L2 projection of fields at quadrature points onto the nodal space of a mesh
with `n_nodes` nodes. Without `colored_matrix` a scalar mass-matrix pattern is built from
`elements` (adapted to `lifter`), and vector/tensor fields are projected component-wise
(multiple right-hand sides); with a colored matrix of `d` dofs per node the coupled system
is solved. The mass matrix and right-hand side are obtained by differentiating the virtual
work `∫ v·u` and `∫ v·field`, as in tatva.
"""
function make_project_function(n_nodes::Integer; colored_matrix = nothing, elements = nothing, lifter = nothing)
	n_nodes = Int(n_nodes)
	if colored_matrix === nothing
		elements === nothing && throw(ArgumentError("Must provide elements if colored_matrix is not provided"))
		dim = 1
		sp = _pattern_from_elements(elements, 1, n_nodes)
		lifter !== nothing && (sp = adapt_sparsity(lifter, sp))
		colored_matrix = ColoredMatrix(sp)
	else
		if lifter !== nothing && size(colored_matrix, 1) != lifter.size_reduced
			throw(ArgumentError("Colored matrix size does not match lifter reduced size. " *
								"Expected $(lifter.size_reduced), got $(size(colored_matrix, 1))"))
		end
		n_full = lifter === nothing ? size(colored_matrix, 1) : lifter.size
		dim = n_full ÷ n_nodes
		dim * n_nodes == n_full || throw(ArgumentError("colored matrix size $n_full is not a multiple of the $n_nodes nodes"))
	end
	lhs, rhs = _make_projection(n_nodes, dim, colored_matrix, lifter)
	n_solver = size(colored_matrix, 1)
	return function project_fn(op::Operator, field::AbstractMatrix)
		# M x = b - r0, with r0 = ∫ v·lift(0): the coupling to non-zero fixed values (tatva
		# drops it, which is only right for homogeneous constraints)
		r0, M = lhs(zeros(n_solver), op)
		b = rhs(field, op)
		b = b isa AbstractVector ? b - r0 : reshape(b, :, n_solver) .- transpose(r0)
		solution = _solve_projection(M, b, lifter)
		value_shape = size(first(field))
		if isempty(value_shape)
			return solution
		elseif solution isa AbstractVector  # coupled solve: interleaved dofs
			return reshape(solution, value_shape..., n_nodes)
		else  # one column per component
			return reshape(permutedims(solution), value_shape..., n_nodes)
		end
	end
end

function _make_projection(n_nodes::Int, dim::Int, colored_matrix::ColoredMatrix, lifter)
	n = lifter === nothing ? n_nodes * max(dim, 1) : lifter.size_reduced
	_solver_field(arr) = dim > 1 ? reshape(arr, dim, :) : arr
	# Galerkin: trial functions u = P u_r + lift(0), test functions v = P v_r (periodic slaves
	# follow their masters, fixed values do not enter), so the system is Pᵀ M P u_r = Pᵀ (b - M lift(0))
	_full_test(test) = lifter === nothing ? test : _lift_linear(lifter, test)
	_full_trial(trial) = lifter === nothing ? trial : lift_from_zeros(lifter, trial)

	function _lhs(test, trial, op)
		v_q = evaluate(op, _solver_field(_full_test(test)))
		u_q = evaluate(op, _solver_field(_full_trial(trial)))
		return integrate(op, _projection_dot.(v_q, u_q))
	end
	function _rhs(test, field, op)
		v_q = evaluate(op, _solver_field(_full_test(test)))
		return integrate(op, _projection_dot.(v_q, field))
	end
	lhs = linearized_jacfwd(virtual_work_to_residual(_lhs; test_size = n), colored_matrix)
	rhs = virtual_work_to_residual(_rhs; test_size = n)
	return lhs, rhs
end

# contracting dot for a coupled solve, component-wise product for a scalar test function
_projection_dot(a::Number, b) = a * b
_projection_dot(a::StaticArray, b::StaticArray) = dot(a, b)
_projection_dot(a::StaticArray, b::Number) = a * b

function _solve_projection(M::ColoredMatrix, b::AbstractArray, lifter)
	A = sparse(M)
	if b isa AbstractVector
		x = spsolve(A, b)
		return lifter === nothing ? x : lift_from_zeros(lifter, x)
	end
	B = permutedims(reshape(b, :, size(b, ndims(b))))  # (n_solver, n_components)
	X = spsolve(A, B)
	lifter === nothing && return X
	return reduce(hcat, [lift_from_zeros(lifter, X[:, k]) for k in axes(X, 2)])
end

#--------------------------------------------------------------------------------
# misc
#--------------------------------------------------------------------------------

"""
	create_g2l(l2g) -> lookup

Inverse of a local-to-global index map: `lookup(global_indices)` returns the local index
of every global index, or `0` if it is not present locally.
"""
function create_g2l(l2g::AbstractVector{<:Integer})
	perm = sortperm(l2g)
	sorted = l2g[perm]
	return function lookup(global_indices)
		map(global_indices) do g
			pos = searchsortedfirst(sorted, g)
			(pos <= length(sorted) && sorted[pos] == g) ? perm[pos] : 0
		end
	end
end
