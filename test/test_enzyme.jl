# The Enzyme backend (reverse-mode gradients) against the ForwardDiff reference. Run in its
# own process by runtests.jl, so that a crash of Enzyme's compiler cannot take the whole
# suite down.

using Test, Tatva, Enzyme, LinearAlgebra, SparseArrays, StaticArrays, ForwardDiff, Random

const mesh_e = unit_square(6, 6)
const op_e = Operator(mesh_e, Tri3())
ψ_e(∇u) = (F = I + ∇u; J = det(F); 0.5 * (tr(F' * F) - 2) - log(J) + 0.75 * log(J)^2)
energy_e(u) = integrate(op_e, ψ_e.(grad(op_e, reshape(u, 2, :))))
energy_e(u, scale) = scale * energy_e(u)

@testset "reverse-mode gradient (2D)" begin
	u = 0.01 .* randn(MersenneTwister(1), 2 * num_nodes(mesh_e))
	@test jacrev(energy_e; backend = AutoEnzyme())(u) ≈ jacrev(energy_e)(u) atol = 1e-12
	@test jacrev(energy_e; backend = AutoEnzyme())(u, 2.0) ≈ 2 .* jacrev(energy_e)(u) atol = 1e-12
end

@testset "reverse-mode gradient (3D, tetrahedra)" begin
	coords = [0.0 1.0 0.0 0.0 1.0; 0.0 0.0 1.0 0.0 1.0; 0.0 0.0 0.0 1.0 1.0]
	op = Operator(Mesh(coords, [1 2; 2 3; 3 4; 4 5]), Tetrahedron4())
	ψ3(∇u) = (F = I + ∇u; J = det(F); 0.5 * (tr(F' * F) - 3) - log(J) + log(J)^2)
	E3(u) = integrate(op, ψ3.(grad(op, reshape(u, 3, :))))
	u = 0.01 .* randn(MersenneTwister(5), 15)
	@test jacrev(E3; backend = AutoEnzyme())(u) ≈ ForwardDiff.gradient(E3, u) atol = 1e-12
end

@testset "array-valued functions and virtual-work residuals" begin
	g(x) = [x[1] * x[2], sin(x[1])]
	@test jacrev(g; backend = AutoEnzyme())([1.0, 2.0]) ≈ ForwardDiff.jacobian(g, [1.0, 2.0])
	n = 9
	x = range(0.0, 1.0; length = n)
	op = Operator(Mesh(permutedims(hcat(x, zeros(n))), permutedims(hcat(1:(n-1), 2:n))), Line2())
	G(w, u) = integrate(op, map((a, b, c, d) -> a * b + 1.5 * d * b + c * d^3, grad(op, w), grad(op, u), evaluate(op, w), evaluate(op, u)))
	u = collect(1.0:n) ./ n
	R_en = virtual_work_to_residual(G; test_size = n, backend = AutoEnzyme())
	@test R_en(u) ≈ virtual_work_to_residual(G; test_size = n)(u) atol = 1e-12
end

@testset "second derivatives are refused with a clear error" begin
	cm = ColoredMatrix(pattern_from_mesh(mesh_e, 2))
	u = zeros(2 * num_nodes(mesh_e))
	@test_throws "not supported" jacfwd(jacrev(energy_e; backend = AutoEnzyme()), cm)(u)
	R = virtual_work_to_residual((w, v) -> sum(w .* v .^ 2); test_size = 3, backend = AutoEnzyme())
	@test_throws "not supported" jacfwd(R, ColoredMatrix(sparse(I, 3, 3)))(ones(3))
end
