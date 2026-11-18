# Large-deformation (neo-Hookean) stretching of a 3D bar with load stepping. ForwardDiff
# differentiates the energy of every tetrahedron (`ElementAssembler`), `EnergyProblem`
# assembles gradient and Hessian straight into the free dofs of the lifter, and `newton`
# minimizes the energy (a Cholesky factorization kept across iterations and load steps, a
# line search). The assembly cost is linear in the mesh size.
#
#     julia --project=. -t auto examples/hyperelasticity.jl

using Tatva, LinearAlgebra, SparseArrays, Printf
include("box_mesh.jl")

const μ, λ = 1.0, 1.5
function ψ(∇u)
	F = I + ∇u
	J = det(F)
	return μ / 2 * (tr(F' * F) - 3) - μ * log(J) + λ / 2 * log(J)^2
end

mesh = box_mesh(32, 8, 8)  # [0, 4] × [0, 1] × [0, 1]
op = Operator(mesh, Tetrahedron4(); cache_weights = true)
Solution = Compound(:Solution, :u => field((3, AUTO)); mesh = mesh)

# clamp x = 0, pull x = 4 along x (lateral motion of that face fixed)
x = mesh.coords[1, :]
left, right = findall(x .== 0.0), findall(x .== 4.0)
lifter = Lifter(Solution.size,
	Fixed(Solution.u[:, left], 0.0),
	Fixed(Solution.u[1, right], RuntimeValue(:stretch, 0.0)),
	Fixed(Solution.u[2:3, right], 0.0),
)

asm = ElementAssembler(op, 3; threaded = true)  # parallel when Julia runs with several threads
density(ξ, Ue, Xe) = ψ(grad(op.element, ξ, Ue, Xe))  # energy density at a quadrature point
problem = EnergyProblem(asm, density, lifter, mesh.coords)  # the energy of the free dofs
@printf("%d tetrahedra, %d free dofs, %d nonzeros, %d threads\n", num_elements(mesh), lifter.size_reduced,
	nnz(problem.pattern), Threads.nthreads())

solver = CholeskySolver()  # kept: one symbolic factorization for the whole computation
u_free = zeros(lifter.size_reduced)
for stretch ∈ range(0.0, 1.0; length = 6)[2:end]  # up to 25% elongation
	result = newton(with_values(problem, :stretch => stretch), u_free; linear_solver = solver)
	@printf("stretch %.2f, residual: %s\n", stretch, join((@sprintf("%.1e", r) for r in result.residuals), " "))
	u_free .= result.u
end
U = reshape(lift_from_zeros(with_values(lifter, :stretch => 1.0), u_free), 3, :)
@printf("lateral contraction at mid-length: %.4f\n", minimum(U[2, findall(isapprox.(x, 2.0))]))
