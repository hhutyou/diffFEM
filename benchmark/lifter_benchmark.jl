# Port of tatva/tests/test_lifter_benchmark.py: the lifter adds no significant overhead to an
# energy evaluation, whether runtime values are changed or not.
#
#     julia --project=. benchmark/lifter_benchmark.jl

using Tatva, LinearAlgebra, Printf, Statistics

const μ, λ = 1 / (2 * 1.3), 0.3 / (1.3 * 0.4)
ψ(∇u) = (ε = (∇u + ∇u') / 2; 0.5 * dot(2μ * ε + λ * tr(ε) * I, ε))

mesh = unit_square(100, 100)
op = Operator(mesh, Tri3())
energy(u_full) = integrate(op, ψ.(grad(op, reshape(u_full, 2, :))))
Solution = Compound(:Solution, :u => field((2, AUTO)); mesh = mesh)
y = mesh.coords[2, :]
bottom, top = findall(y .== 0), findall(y .== 1)
lifter = Lifter(Solution.size,
	Fixed(Solution.u[:, bottom], RuntimeValue("bottom", 0.0)),
	Fixed(Solution.u[2, top], RuntimeValue("top", 0.5)),
)
top_dofs = Solution.u[2, top]

energy_lifted(u_free, lf) = energy(lift_from_zeros(lf, u_free))
energy_disp_top(u_free, disp) = energy(lift_from_zeros(lifter.at["top"].set(disp), u_free))
function energy_manual(u_free, disp)
	u = zeros(eltype(u_free), lifter.size)
	u[lifter.free_dofs] = u_free
	u[top_dofs] .= disp
	return energy(u)
end

function timed(f, args...; runs = 20)
	t0 = time()
	f(args...)
	first = time() - t0
	return first, median([(t0 = time(); f(args...); time() - t0) for _ in 1:runs])
end

u_free = zeros(lifter.size_reduced)
@printf("Benchmarking lifted energies [n_dofs = %d]\n", lifter.size)
for (name, f, arg) ∈ (
	("lifter", energy_lifted, lifter),
	("lifter, new value", energy_lifted, with_values(lifter, "top" => 0.7)),
	("disp_top", energy_disp_top, 0.5),
	("manual", energy_manual, 0.5),
)
	first, mean = timed(f, u_free, arg)
	@printf("%20s: first = %.6f s, median = %.6f s, energy = %.6e\n", name, first, mean, f(u_free, arg))
end
