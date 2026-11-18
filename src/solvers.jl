#--------------------------------------------------------------------------------
# Solvers (a Julia addition to tatva's API): the element-wise energy as a function of the
# free dofs, linear solvers that reuse their factorizations, Newton's method
#--------------------------------------------------------------------------------

"""
	EnergyProblem(asm, density, lifter, extras...; field_shape)

The energy of [`ElementAssembler`](@ref) `asm` as a function of the free dofs of `lifter`:
`E(u_free) = Σₑ Σ_q w_q detJ_q density(ξ_q, Uₑ, extrasₑ...)` with the nodal field
`U = reshape(lift_from_zeros(lifter, u_free), field_shape..., :)` (by default a vector for one
component per node, `(n_comp, n_nodes)` otherwise). Gradient and Hessian are assembled
directly into the free dofs (`Pᵀ g` and `Pᵀ K P`, without forming the full ones); `Fixed` and
`Periodic` constraints are supported.

Evaluate with [`assemble`](@ref), [`assemble!`](@ref), [`assemble_energy`](@ref) and
[`assemble_gradient`](@ref) at `u_free`, minimize with [`newton`](@ref).
`with_values(problem, key => value)` changes runtime values of the lifter (load steps) and
keeps the structure.

	problem = EnergyProblem(asm, density, lifter, mesh.coords)
	result = newton(problem, zeros(lifter.size_reduced))
"""
struct EnergyProblem{A <: ElementAssembler, D, L <: Lifter, X <: Tuple, S}
	assembler::A
	density::D
	lifter::L
	extras::X
	field_shape::S                          # leading axes of the nodal field
	dofs::Matrix{Int}                       # free dof of every element dof (0: constrained)
	positions::Matrix{Int}                  # nzval index in `pattern` of every element pair (0: dropped)
	pattern::SparseMatrixCSC{Float64, Int}  # structure of the Hessian of the free dofs
	groups::Vector{Vector{UnitRange{Int}}}  # element blocks of the threaded loops, by free dofs
end

function EnergyProblem(asm::ElementAssembler{O, NC}, density, lifter::Lifter, extras...;
	field_shape::Tuple = NC == 1 ? () : (NC,)) where {O, NC}
	n = size(asm.pattern, 1)
	lifter.size == n && lifter.nb_extra_ghost_dofs === nothing ||
		throw(DimensionMismatch("the lifter has $(lifter.size) dofs, the assembler $n"))
	prod(field_shape; init = 1) == NC || throw(DimensionMismatch("field_shape $field_shape does not have $NC components"))
	src = _lift_sources(lifter)
	src === nothing && throw(ArgumentError("EnergyProblem supports Fixed and Periodic constraints (use assemble with reduce_hessian otherwise)"))
	dofs = map(d -> src[d], asm.dofs)
	nloc = size(dofs, 1)
	I, J = Int[], Int[]
	for e ∈ axes(dofs, 2), b ∈ 1:nloc, a ∈ 1:nloc
		dofs[a, e] > 0 && dofs[b, e] > 0 || continue
		push!(I, dofs[a, e])
		push!(J, dofs[b, e])
	end
	m = lifter.size_reduced
	pattern = sparse(I, J, zeros(length(I)), m, m)
	positions = zeros(Int, nloc * nloc, size(dofs, 2))
	rv = rowvals(pattern)
	for e ∈ axes(dofs, 2), b ∈ 1:nloc
		dofs[b, e] > 0 || continue
		r = nzrange(pattern, dofs[b, e])
		for a ∈ 1:nloc
			dofs[a, e] > 0 && (positions[(b-1)*nloc+a, e] = first(r) + searchsortedfirst(view(rv, r), dofs[a, e]) - 1)
		end
	end
	return EnergyProblem(asm, density, lifter, extras, field_shape, dofs, positions, pattern, _element_groups(dofs))
end

Base.show(io::IO, p::EnergyProblem) =
	print(io, "EnergyProblem(", p.lifter.size_reduced, " free dofs, nnz=", nnz(p.pattern), ", ", p.assembler, ")")

with_values(p::EnergyProblem, updates...) = EnergyProblem(p.assembler, p.density, with_values(p.lifter, updates...), p.extras,
	p.field_shape, p.dofs, p.positions, p.pattern, p.groups)

# the nodal field of the free dofs
_field(p::EnergyProblem, u_free::AbstractVector) = reshape(lift_from_zeros(p.lifter, u_free), p.field_shape..., :)

function _check_free(p::EnergyProblem, u_free::AbstractVector)
	length(u_free) == p.lifter.size_reduced ||
		throw(DimensionMismatch("the problem has $(p.lifter.size_reduced) free dofs, got a vector of length $(length(u_free))"))
	return u_free
end

assemble_energy(p::EnergyProblem, u_free::AbstractVector) =
	assemble_energy(p.assembler, p.density, _field(p, _check_free(p, u_free)), p.extras...)

assemble_gradient(p::EnergyProblem, u_free::AbstractVector) = _assemble_gradient(nothing, p, u_free)
assemble_gradient!(g::AbstractVector, p::EnergyProblem, u_free::AbstractVector) =
	_assemble_gradient(_check_free(p, g), p, u_free)

function _assemble_gradient(g, p::EnergyProblem, u_free::AbstractVector)
	asm = p.assembler
	data = _assembly_data(asm, _field(p, _check_free(p, u_free)), p.extras)
	f1, x1 = _element_function(asm, p.density, data..., 1)
	first_element = _value_gradient(f1, x1)
	g = g === nothing ? zeros(eltype(last(first_element)), size(p.pattern, 1)) : fill!(g, zero(eltype(g)))
	_assemble_gradient!(g, asm, p.density, data, first_element, p.dofs, p.groups)
	return g
end

function assemble(p::EnergyProblem, u_free::AbstractVector)
	asm = p.assembler
	data = _assembly_data(asm, _field(p, _check_free(p, u_free)), p.extras)
	f1, x1 = _element_function(asm, p.density, data..., 1)
	first_element = _value_gradient_hessian(f1, x1)
	T = typeof(first(first_element))
	P = p.pattern
	g = zeros(T, size(P, 1))
	K = SparseMatrixCSC(P.m, P.n, copy(P.colptr), copy(P.rowval), zeros(T, nnz(P)))
	E = _assemble!(g, nonzeros(K), asm, p.density, data, first_element, p.dofs, p.positions, p.groups)
	return E, g, K
end

function assemble!(g::AbstractVector, K::SparseMatrixCSC, p::EnergyProblem, u_free::AbstractVector)
	_check_free(p, g)
	P = p.pattern
	size(K) == size(P) && K.colptr == P.colptr && K.rowval == P.rowval ||
		throw(ArgumentError("the Hessian must have the structure of the problem's pattern (e.g. similar(problem.pattern))"))
	asm = p.assembler
	data = _assembly_data(asm, _field(p, _check_free(p, u_free)), p.extras)
	f1, x1 = _element_function(asm, p.density, data..., 1)
	fill!(g, zero(eltype(g)))
	fill!(nonzeros(K), zero(eltype(K)))
	return _assemble!(g, nonzeros(K), asm, p.density, data, _value_gradient_hessian(f1, x1), p.dofs, p.positions, p.groups)
end

#--------------------------------------------------------------------------------
# linear solvers
#--------------------------------------------------------------------------------

"""
	LinearSolver

Solver of the linear systems of [`newton`](@ref): `solve!(solver, K, b)` returns `x` with
`K x = b`. A solver keeps its factorization between calls: while the sparsity pattern of `K`
stays the same, only the numerical factorization is redone (the symbolic analysis, ordering
included, is reused). Built in: [`DirectSolver`](@ref) (sparse LU) and
[`CholeskySolver`](@ref); with `using LinearSolve`, [`LinearSolveSolver`](@ref) runs any
LinearSolve.jl algorithm.
"""
abstract type LinearSolver end

"""
	DirectSolver()

Sparse LU factorization (UMFPACK) for any square matrix; see [`LinearSolver`](@ref).
"""
mutable struct DirectSolver <: LinearSolver
	factorization::Any
	pattern::Tuple{Vector{Int}, Vector{Int}}  # colptr and rowval of the factorized matrix
end
DirectSolver() = DirectSolver(nothing, (Int[], Int[]))

"""
	CholeskySolver()

Sparse Cholesky factorization (CHOLMOD) of the upper triangle of a symmetric matrix, about
twice as fast as LU; matrices that are not positive definite (an unstable state, a saddle
point) are solved by LU instead. See [`LinearSolver`](@ref).
"""
mutable struct CholeskySolver <: LinearSolver
	factorization::Any
	pattern::Tuple{Vector{Int}, Vector{Int}}
	fallback::DirectSolver
end
CholeskySolver() = CholeskySolver(nothing, (Int[], Int[]), DirectSolver())

"""
	LinearSolveSolver(alg; preconditioner=nothing, kwargs...)

A [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl) algorithm as a
[`LinearSolver`](@ref) (requires `using LinearSolve`): e.g. `KLUFactorization()` or
`KrylovJL_CG()` (conjugate gradients need positive definite Hessians; `KrylovJL_MINRES()`
does not). `preconditioner(K)`, if given, builds the left preconditioner whenever the
matrix changes — an algebraic multigrid one for instance, with the near null space from
[`rigid_body_modes`](@ref). `kwargs` (`reltol`, `maxiters`, …) go to `LinearSolve.init`. The
LinearSolve cache is kept between calls.
"""
mutable struct LinearSolveSolver{A, P, K} <: LinearSolver
	alg::A
	preconditioner::P
	kwargs::K
	cache::Any
end
LinearSolveSolver(alg; preconditioner = nothing, kwargs...) = LinearSolveSolver(alg, preconditioner, kwargs, nothing)

_same_pattern(pattern, K::SparseMatrixCSC) = pattern[1] == K.colptr && pattern[2] == K.rowval

"""
	solve!(solver::LinearSolver, K, b) -> x

Solve `K x = b`, refactorizing `K` (see [`LinearSolver`](@ref)).
"""
function solve!(s::DirectSolver, K::SparseMatrixCSC{<:Union{Float64, ComplexF64}}, b::AbstractVector)
	if s.factorization !== nothing && _same_pattern(s.pattern, K)
		lu!(s.factorization, K)
	else
		s.factorization = lu(K)
		s.pattern = (copy(K.colptr), copy(K.rowval))
	end
	return s.factorization \ b
end

function solve!(s::CholeskySolver, K::SparseMatrixCSC{Float64}, b::AbstractVector)
	A = Symmetric(K, :U)
	if s.factorization !== nothing && _same_pattern(s.pattern, K)
		cholesky!(s.factorization, A; check = false)
	else
		s.factorization = cholesky(A; check = false)
		s.pattern = (copy(K.colptr), copy(K.rowval))
	end
	issuccess(s.factorization) && return s.factorization \ b
	return solve!(s.fallback, K, b)  # not positive definite
end

solve!(s::LinearSolveSolver, K::AbstractMatrix, b::AbstractVector) = _linearsolve_solve!(s, K, b)
_linearsolve_solve!(_, _, _) = throw(ArgumentError("LinearSolveSolver requires `using LinearSolve`"))

# other number types (dual numbers, …): no factorization to keep
solve!(::LinearSolver, K::AbstractMatrix, b::AbstractVector) = spsolve(sparse(K), b)

"""
	rigid_body_modes(coords; lifter=nothing) -> Matrix

The rigid body motions of a body with nodal coordinates `coords` (`dim × n_nodes`, `dim` = 2
or 3) as columns of dof vectors (`dim` interleaved components per node): `dim` translations
and the rotations (3 modes in 2D, 6 in 3D). They span the null space of the elasticity
operator of a free body, the "near null space" that algebraic multigrid preconditioners need
for elasticity. With a `lifter`, restricted to its free dofs.
"""
function rigid_body_modes(coords::AbstractMatrix; lifter::Union{Nothing, Lifter} = nothing)
	d, n = size(coords)
	d in (2, 3) || throw(ArgumentError("rigid body modes are defined for 2 or 3 dimensions, got $d"))
	B = zeros(d * n, d == 2 ? 3 : 6)
	for k ∈ 1:n
		r = d * (k - 1)
		for c ∈ 1:d
			B[r+c, c] = 1.0  # translations
		end
		x, y = coords[1, k], coords[2, k]
		if d == 2
			B[r+1, 3], B[r+2, 3] = -y, x
		else
			z = coords[3, k]
			B[r+2, 4], B[r+3, 4] = -z, y   # rotation about x
			B[r+1, 5], B[r+3, 5] = z, -x   # about y
			B[r+1, 6], B[r+2, 6] = -y, x   # about z
		end
	end
	lifter === nothing && return B
	size(B, 1) == lifter.size || throw(DimensionMismatch("the lifter has $(lifter.size) dofs, the modes $(size(B, 1))"))
	return B[lifter.free_dofs, :]
end

#--------------------------------------------------------------------------------
# Newton's method
#--------------------------------------------------------------------------------

"""Outcome of [`newton`](@ref)."""
struct NewtonResult{V}
	u::V                        # the free dofs
	converged::Bool
	iterations::Int
	residuals::Vector{Float64}  # ‖g‖ at every iterate (the first: the initial guess)
	energies::Vector{Float64}
	steps::Vector{Float64}      # line search step length of every iteration
end

Base.show(io::IO, r::NewtonResult) = print(io, "NewtonResult(", r.converged ? "converged" : "not converged", " after ",
	r.iterations, " iteration", r.iterations == 1 ? "" : "s", ", ‖g‖ = ", r.residuals[end], ")")

"""
	newton(problem::EnergyProblem, u0; linear_solver=CholeskySolver(), atol=1e-10, rtol=1e-10,
	       maxiter=25, linesearch=true, verbose=false) -> NewtonResult

Minimize the energy of `problem` over the free dofs from `u0` (or solve `∇E = 0`) by
Newton's method. Each iteration assembles energy, gradient `g` and Hessian `K` in one pass
and solves `K Δu = g` with `linear_solver` (by default a Cholesky factorization, which falls
back to LU where `K` is not positive definite); keep the solver across load steps to reuse its
factorization. With `linesearch` the method minimizes: a step that does not decrease the
energy (`K` not positive definite) is replaced by the step of `K + τI` with the smallest
positive definite shift `τ` tried (modified Newton), and the step is shortened by
backtracking until the energy decreases enough (Armijo), also where the energy cannot be
evaluated at the full step (an element turned inside out, the `log` of a negative Jacobian,
…). Without it, plain Newton steps (saddle points, root finding). The energy must be
defined at `u0`: apply large loads in steps. Converged when `‖g‖ ≤ max(atol, rtol ‖g₀‖)`.
"""
function newton(p::EnergyProblem, u0::AbstractVector; linear_solver::LinearSolver = CholeskySolver(), atol::Real = 1e-10,
	rtol::Real = 1e-10, maxiter::Integer = 25, linesearch::Bool = true, verbose::Bool = false)
	u = Vector{Float64}(_check_free(p, u0))
	g = zeros(length(u))
	K = similar(p.pattern)
	E = assemble!(g, K, p, u)
	residuals, energies, steps = [norm(g)], [Float64(E)], Float64[]
	tol = max(atol, rtol * residuals[1])
	shifted = Ref{Any}(nothing)  # factorization of K + τI, see _shifted_step
	verbose && println("newton: ‖g‖ = ", residuals[1], ", E = ", E)
	for it ∈ 1:maxiter
		residuals[end] <= tol && return NewtonResult(u, true, it - 1, residuals, energies, steps)
		Δ = solve!(linear_solver, K, g)
		linesearch && !(dot(g, Δ) > 0) && (Δ = _shifted_step(shifted, K, g))  # not a descent direction
		α = linesearch ? _backtrack(p, u, Δ, E, g) : 1.0
		α > 0 || return NewtonResult(u, false, it, residuals, energies, push!(steps, 0.0))  # no acceptable step
		u .-= α .* Δ
		E = assemble!(g, K, p, u)
		push!(residuals, norm(g))
		push!(energies, E)
		push!(steps, α)
		verbose && println("newton: iteration ", it, ", step ", α, ", ‖g‖ = ", residuals[end], ", E = ", E)
	end
	return NewtonResult(u, residuals[end] <= tol, maxiter, residuals, energies, steps)
end

# Where K is not positive definite the Newton step need not decrease the energy: the step of
# the positive definite K + τI instead (modified Newton), τ growing tenfold from 1e-8 of the
# mean diagonal until the Cholesky factorization succeeds (CHOLMOD shifts the matrix itself).
function _shifted_step(shifted::Ref{Any}, K::SparseMatrixCSC{Float64}, g)
	A = Symmetric(K, :U)
	τ = 1e-8 * max(sum(abs, diag(K)) / size(K, 1), floatmin())
	for _ ∈ 1:24
		F = shifted[] === nothing ? cholesky(A; shift = τ, check = false) : cholesky!(shifted[], A; shift = τ, check = false)
		shifted[] = F
		issuccess(F) && return F \ g
		τ *= 10
	end
	return g  # steepest descent
end
_shifted_step(::Ref{Any}, ::AbstractMatrix, g) = g

# Step length α ∈ (0, 1] for u - α Δ, halving the step until the energy decreases enough
# (Armijo); 0 if no step qualifies. Where that test is meaningless — not a descent direction
# (an indefinite Hessian), or a decrease below the resolution of the energy (the quadratic
# convergence regime) — a step only needs an energy that can be evaluated.
function _backtrack(p::EnergyProblem, u, Δ, E, g; c = 1e-4, halvings = 30)
	slope = dot(g, Δ)  # the decrease rate of E(u - α Δ) at α = 0
	armijo = slope > 1e-8 * abs(E)
	α = 1.0
	for _ ∈ 0:halvings
		Et = _energy_or_inf(p, u .- α .* Δ)
		(armijo ? Et <= E - c * α * slope : isfinite(Et)) && return α
		α /= 2
	end
	return 0.0
end

# the energy, Inf where it cannot be evaluated
function _energy_or_inf(p::EnergyProblem, u)
	E = try
		assemble_energy(p, u)
	catch err
		_is_domain_error(err) || rethrow()
		return Inf
	end
	return isfinite(E) ? E : Inf
end

_is_domain_error(err) = err isa DomainError
_is_domain_error(err::TaskFailedException) = _is_domain_error(err.task.exception)
_is_domain_error(err::CompositeException) = all(_is_domain_error, err.exceptions)
