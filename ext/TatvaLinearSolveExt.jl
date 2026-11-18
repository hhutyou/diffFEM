# LinearSolve.jl algorithms as linear solvers of `newton`, loaded with `using LinearSolve`
# (see `Tatva.LinearSolveSolver`).

module TatvaLinearSolveExt

using Tatva
using Tatva: LinearSolveSolver
using LinearSolve
using SparseArrays

function Tatva._linearsolve_solve!(s::LinearSolveSolver, K::AbstractMatrix, b::AbstractVector)
	Pl = s.preconditioner === nothing ? nothing : s.preconditioner(K)
	cache = s.cache
	if cache === nothing || !_same_structure(cache.A, K)
		kwargs = Pl === nothing ? (; s.kwargs...) : (; s.kwargs..., Pl = Pl)
		s.cache = cache = LinearSolve.init(LinearProblem(K, b), s.alg; kwargs...)
	else  # a new matrix of the same structure: refactorize (reusing the symbolic analysis)
		cache.A = K
		cache.b = b
		Pl === nothing || (cache.Pl = Pl)
	end
	return copy(LinearSolve.solve!(cache).u)
end

_same_structure(A::SparseMatrixCSC, K::SparseMatrixCSC) = size(A) == size(K) && A.colptr == K.colptr && A.rowval == K.rowval
_same_structure(A, K) = size(A) == size(K) && typeof(A) == typeof(K)

end # module
