# Port of tatva/tests/test_sparse_benchmark.py: the sparse Hessian of a 2D linear-elastic
# energy, with and without an auxiliary argument (a damage field) — the timings should be
# about the same, i.e. extra arguments do not trigger re-tracing or slow paths.
#
#     julia --project=. benchmark/sparse_benchmark.jl
#
# (The global forward-mode path grows like n_dofs², so meshes are smaller than tatva's
# 100×100 / 200×200; see colored_matrix_benchmark.jl for the element-wise path.)

using Tatva, LinearAlgebra, Printf, Statistics

const E_young, ν = 1.0, 0.3
const μ = E_young / (2 * (1 + ν))
const λ = E_young * ν / ((1 + ν) * (1 - 2ν))
ψ(∇u) = (ε = (∇u + ∇u') / 2; 0.5 * dot(2μ * ε + λ * tr(ε) * I, ε))

for n ∈ (10, 20, 30)
	mesh = unit_square(n, n)
	op = Operator(mesh, Tri3())
	total_energy(u) = integrate(op, ψ.(grad(op, reshape(u, 2, :))))
	total_energy_with_args(u, damage) = total_energy(u) * sum(damage) / length(damage)
	cm = ColoredMatrix(pattern_from_mesh(mesh, 2))
	hessian = jacfwd(jacrev(total_energy), cm; color_batch_size = 10)
	hessian_with_args = jacfwd(jacrev(total_energy_with_args), cm; color_batch_size = 10)
	u = zeros(2 * num_nodes(mesh))
	damage = ones(num_nodes(mesh))

	hessian(u)
	t_plain = median([@elapsed(hessian(u)) for _ in 1:3])
	hessian_with_args(u, damage)
	t_args = median([@elapsed(hessian_with_args(u, damage)) for _ in 1:3])
	@printf("n_dofs = %5d: without args %.4f s, with args %.4f s\n", length(u), t_plain, t_args)
end
