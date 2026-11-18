# Port of tatva/tests/test_colored_matrix_benchmark.py, extended to compare the ways of
# differentiating a 2D linear-elastic energy ("colors": those of the sparse Hessian):
#
#   global ForwardDiff   jacrev(E) / sparse_hessian(E, cm)  (tatva's API; O(n²) in Julia)
#   global Enzyme        jacrev(E; backend = AutoEnzyme())   (reverse mode; gradient only)
#   element-wise         ElementAssembler                    (energy, gradient and Hessian)
#
#     julia --project=. -t auto benchmark/colored_matrix_benchmark.jl
# (Enzyme must be installed in the active environment for the Enzyme column; the last column
# is the threaded element-wise assembly when Julia runs with several threads.)

using Tatva, LinearAlgebra, SparseArrays, Printf, Statistics
const HAS_ENZYME = Base.find_package("Enzyme") !== nothing
HAS_ENZYME && @eval using Enzyme

const μ, λ = 1.0, 1.0
ψ(∇u) = (ε = (∇u + ∇u') / 2; μ * dot(ε, ε) + λ / 2 * tr(ε)^2)

timed(f) = (f(); median([(t0 = time_ns(); f(); (time_ns() - t0) / 1e9) for _ in 1:5]))
ms(t) = t === nothing ? "       -" : @sprintf("%8.2f", 1e3 * t)

@printf("%9s %8s %6s | %25s | %10s | %25s | %s\n", "mesh", "n_dofs", "colors", "global FD: grad / hess [ms]",
	"Enzyme grad", "element-wise: grad / all [ms]", "all, $(Threads.nthreads()) threads")
for n ∈ (10, 20, 50, 100, 200)
	mesh = unit_square(n, n)
	op = Operator(mesh, Tri3())
	E(u) = integrate(op, ψ.(grad(op, reshape(u, 2, :))))
	u = 0.01 .* randn(2 * num_nodes(mesh))
	cm = ColoredMatrix(pattern_from_mesh(mesh, 2))

	small = length(u) <= 1000
	t_fd_g = small ? timed(() -> jacrev(E)(u)) : nothing
	t_fd_h = small ? timed(() -> sparse_hessian(E, cm)(u)) : nothing
	t_en = HAS_ENZYME ? timed(() -> jacrev(E; backend = AutoEnzyme())(u)) : nothing

	asm = ElementAssembler(op, 2)
	density(ξ, Ue, Xe) = ψ(grad(op.element, ξ, Ue, Xe))
	U = reshape(u, 2, :)
	t_el_g = timed(() -> assemble_gradient(asm, density, U, mesh.coords))
	t_el_all = timed(() -> assemble(asm, density, U, mesh.coords))
	asm_t = ElementAssembler(op, 2; threaded = true)
	t_el_t = Threads.nthreads() > 1 ? timed(() -> assemble(asm_t, density, U, mesh.coords)) : nothing

	@printf("%4d×%-4d %8d %6d | %12s / %12s | %10s | %12s / %12s | %12s\n", n, n, length(u), n_colors(sparse_hessian(E, cm)),
		ms(t_fd_g), ms(t_fd_h), ms(t_en), ms(t_el_g), ms(t_el_all), ms(t_el_t))
end
