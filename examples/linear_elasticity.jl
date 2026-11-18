# Linear elasticity of a plate pulled at its right edge — the energy-based workflow of
# tatva in Julia: write the strain energy, let automatic differentiation provide the
# residual and the sparse tangent, solve.
#
#     julia --project=. examples/linear_elasticity.jl

using Tatva, LinearAlgebra, SparseArrays, Printf

# material (plane strain)
const E, ν = 1.0, 0.3
const μ = E / (2 * (1 + ν))
const λ = E * ν / ((1 + ν) * (1 - 2ν))

# strain energy density of the displacement gradient ∇u (an SMatrix{2,2})
function ψ(∇u)
	ε = (∇u + ∇u') / 2
	return μ * dot(ε, ε) + λ / 2 * tr(ε)^2
end

mesh = rectangle((0.0, 2.0), (0.0, 1.0), 20, 10)  # global AD: keep it small (see ElementAssembler)
op = Operator(mesh, Tri3())

# the unknowns: a 2-component nodal field u, stored node by node in one flat vector
Solution = Compound(:Solution, :u => field((2, AUTO)); mesh = mesh)

# boundary conditions: clamp the left edge, pull the right edge by `stretch` in x
x = mesh.coords[1, :]
left, right = findall(x .== 0.0), findall(x .== 2.0)
lifter = Lifter(Solution.size,
	Fixed(Solution.u[:, left], 0.0),
	Fixed(Solution.u[1, right], RuntimeValue(:stretch, 0.0)),
)

# total energy as a function of the free dofs
energy(u_free, lf) = integrate(op, ψ.(grad(op, Solution(lift_from_zeros(lf, u_free)).u)))

# sparse tangent: pattern from the mesh, reduced to the free dofs, colored for AD
pattern = adapt_sparsity(lifter, pattern_from_mesh(mesh, 2))
cm = ColoredMatrix(pattern)
@printf("free dofs: %d, nonzeros: %d, colors: %d\n", lifter.size_reduced, nnz(cm), n_colors(cm))

residual = jacrev(energy)          # ∂E/∂u
tangent = sparse_hessian(energy, cm)  # ∂²E/∂u², assembled by colored forward-mode AD

lf = with_values(lifter, :stretch => 0.02)
u_free = zeros(lifter.size_reduced)
@time K = tangent(u_free, lf)
@time r = residual(u_free, lf)
u_free -= spsolve(K, r)  # quadratic energy: one Newton step
@printf("residual after the solve: %.2e\n", norm(residual(u_free, lf)))

u = Solution(lift_from_zeros(lf, u_free)).u
top_middle = findfirst(n -> isapprox(x[n], 1.0) && mesh.coords[2, n] == 1.0, 1:num_nodes(mesh))
@printf("vertical displacement of the top edge at x = 1: %.5f (lateral contraction)\n", u[2, top_middle])
@printf("stored energy: %.6e\n", energy(u_free, lf))
