# EnergyProblem (assembly into the free dofs), linear solvers that keep their factorization,
# Newton's method with an energy line search, rigid body modes.

using Test, Tatva, LinearAlgebra, SparseArrays, StaticArrays, ForwardDiff, Random

const μ, λ = 1.0, 4.0
ψ_neo(∇u) = (F = I + ∇u; J = det(F); μ / 2 * (tr(F' * F) - size(F, 1)) - μ * log(J) + λ / 2 * log(J)^2)
ψ_lin(∇u) = (ε = (∇u + ∇u') / 2; μ * dot(ε, ε) + λ / 2 * tr(ε)^2)
dofs(c, nodes) = 2 .* (nodes .- 1) .+ c

function plate(n; threaded = false)
	mesh = unit_square(n, n)
	op = Operator(mesh, Tri3())
	density(ξ, Ue, Xe) = ψ_neo(grad(op.element, ξ, Ue, Xe))
	x = mesh.coords[1, :]
	return mesh, ElementAssembler(op, 2; threaded), density, findall(x .== 0), findall(x .== 1)
end

@testset "EnergyProblem: assembly into the free dofs = full assembly + reduction" begin
	mesh, asm, density, left, right = plate(5)
	n = 2 * num_nodes(mesh)
	bottom = findall(mesh.coords[2, :] .== 0)
	lifters = (
		Lifter(n, Fixed(vcat(dofs(1, left), dofs(2, left)), 0.0), Fixed(dofs(1, right), 0.1)),
		Lifter(n, Fixed(dofs(1, left), 0.0), Fixed(dofs(2, bottom[1:1]), 0.0), Periodic(dofs(2, right), dofs(2, left))),
		# chained: the right edge follows the left one, whose y dofs follow the bottom-left node
		Lifter(n, Fixed(dofs(1, left), 0.0), Periodic(dofs(2, left[2:end]), fill(dofs(2, left[1:1])[1], length(left) - 1)),
			Periodic(dofs(1, right), dofs(1, left)), Fixed(dofs(2, left[1:1]), 0.02)),
	)
	for lifter ∈ lifters, threaded ∈ (false, true)
		asm_t = ElementAssembler(asm.op, 2; threaded)
		p = EnergyProblem(asm_t, density, lifter, mesh.coords)
		ur = 0.01 .* cos.(1:lifter.size_reduced)
		E, g, K = assemble(p, ur)
		Ef, gf, Kf = assemble(asm, density, reshape(lift_from_zeros(lifter, ur), 2, :), mesh.coords)
		@test E ≈ Ef
		@test g ≈ reduce_adjoint(lifter, gf)
		@test K ≈ reduce_hessian(lifter, Kf)
		@test assemble_energy(p, ur) ≈ E
		@test assemble_gradient(p, ur) ≈ g
		g2, K2 = fill(NaN, length(g)), similar(p.pattern)
		@test assemble!(g2, K2, p, ur) ≈ E
		@test g2 ≈ g && K2 ≈ K
		@test assemble_gradient!(fill(NaN, length(g)), p, ur) ≈ g
	end
	p = EnergyProblem(asm, density, lifters[1], mesh.coords)
	@test_throws DimensionMismatch assemble(p, zeros(3))
	@test_throws ArgumentError assemble!(zeros(lifters[1].size_reduced), similar(asm.pattern), p, zeros(lifters[1].size_reduced))
	@test_throws DimensionMismatch EnergyProblem(asm, density, Lifter(n + 2), mesh.coords)
	@test_throws DimensionMismatch EnergyProblem(asm, density, lifters[1], mesh.coords; field_shape = (3,))
	struct Scale <: Tatva.Constraint
		dofs::Vector{Int}
		id::Tatva.UUID
	end
	@test_throws ArgumentError EnergyProblem(asm, density, Lifter(n, Scale([3], Tatva.uuid4())), mesh.coords)
end

@testset "linear solvers keep their factorization" begin
	A = sprandn(MersenneTwister(1), 60, 60, 0.1)
	K = A' * A + 5I  # symmetric positive definite
	b = randn(MersenneTwister(2), 60)
	for solver ∈ (DirectSolver(), CholeskySolver())
		@test Tatva.solve!(solver, K, b) ≈ K \ b
		K2 = copy(K)
		nonzeros(K2) .*= 2  # same pattern: numerical refactorization
		@test Tatva.solve!(solver, K2, b) ≈ K2 \ b
		K3 = K + sparse(1:60, 60:-1:1, 0.5)  # new pattern: new factorization
		@test Tatva.solve!(solver, K3, b) ≈ K3 \ b
	end
	# not positive definite: Cholesky falls back to LU
	Kind = K - 20I
	@test Tatva.solve!(CholeskySolver(), Kind, b) ≈ Kind \ b
	# other number types are solved without a kept factorization
	@test Tatva.solve!(DirectSolver(), sparse(BigFloat.(Matrix(K))), BigFloat.(b)) ≈ K \ b
end

@testset "newton: quadratic convergence, load steps, solver reuse" begin
	mesh, asm, density, left, right = plate(8)
	n = 2 * num_nodes(mesh)
	lifter = Lifter(n, Fixed(vcat(dofs(1, left), dofs(2, left)), 0.0), Fixed(dofs(1, right), RuntimeValue(:stretch, 0.0)))
	p = EnergyProblem(asm, density, lifter, mesh.coords)
	solver = CholeskySolver()
	u = zeros(lifter.size_reduced)
	for stretch ∈ (0.1, 0.2)
		r = newton(with_values(p, :stretch => stretch), u; linear_solver = solver)
		@test r.converged
		@test r.residuals[end] < 1e-10
		@test all(r.steps .== 1.0)  # small load steps: full Newton steps
		@test r.iterations <= 6     # quadratic convergence
		u = r.u
	end
	# the same solution with LU and with the global AD gradient
	r_lu = newton(with_values(p, :stretch => 0.2), zeros(lifter.size_reduced); linear_solver = DirectSolver())
	@test r_lu.u ≈ u
	E_global(ur) = integrate(asm.op, ψ_neo.(grad(asm.op, reshape(lift_from_zeros(with_values(lifter, :stretch => 0.2), ur), 2, :))))
	@test norm(jacrev(E_global)(u)) < 1e-9
	@test sprint(show, r_lu) == "NewtonResult(converged after $(r_lu.iterations) iterations, ‖g‖ = $(r_lu.residuals[end]))"
end

@testset "newton: the line search rescues a large step" begin
	mesh, asm, density, left, right = plate(8)
	n = 2 * num_nodes(mesh)
	lifter = Lifter(n, Fixed(vcat(dofs(1, left), dofs(2, left)), 0.0), Fixed(dofs(1, right), 2.0), Fixed(dofs(2, right), 0.0))
	p = EnergyProblem(asm, density, lifter, mesh.coords)
	@test_throws DomainError newton(p, zeros(lifter.size_reduced); linesearch = false)  # an element turns inside out
	r = newton(p, zeros(lifter.size_reduced); maxiter = 40)
	@test r.converged
	@test any(<(1), r.steps)
	@test all(diff(r.energies) .<= 1e-12 * abs(r.energies[1]))  # every step decreases the energy
	# the threaded assembler finds the same equilibrium
	r_t = newton(EnergyProblem(ElementAssembler(asm.op, 2; threaded = true), density, lifter, mesh.coords), zeros(lifter.size_reduced); maxiter = 40)
	@test r_t.u ≈ r.u
end

@testset "rigid body modes span the null space of elasticity" begin
	for (mesh, el, d) ∈ ((unit_square(4, 3), Tri3(), 2),
		(Mesh([0.0 1.0 0.0 0.0 1.0; 0.0 0.0 1.0 0.0 1.0; 0.0 0.0 0.0 1.0 1.0], [1 2; 2 3; 3 4; 4 5]), Tetrahedron4(), 3))
		op = Operator(mesh, el)
		density(ξ, Ue, Xe) = ψ_lin(grad(op.element, ξ, Ue, Xe))
		K = assemble_hessian(ElementAssembler(op, d), density, zeros(d, num_nodes(mesh)), mesh.coords)
		B = rigid_body_modes(mesh.coords)
		@test size(B) == (d * num_nodes(mesh), d == 2 ? 3 : 6)
		@test rank(B) == size(B, 2)
		@test norm(K * B) < 1e-12 * norm(K)
	end
	mesh = unit_square(2, 2)
	lifter = Lifter(2 * num_nodes(mesh), Fixed([1, 2]))
	@test size(rigid_body_modes(mesh.coords; lifter)) == (lifter.size_reduced, 3)
	@test_throws ArgumentError rigid_body_modes(zeros(1, 3))
end

using LinearSolve, AlgebraicMultigrid

@testset "LinearSolve.jl algorithms as linear solvers" begin
	mesh, asm, density, left, right = plate(10)
	n = 2 * num_nodes(mesh)
	lifter = Lifter(n, Fixed(vcat(dofs(1, left), dofs(2, left)), 0.0), Fixed(dofs(1, right), RuntimeValue(:stretch, 0.2)))
	p = EnergyProblem(asm, density, lifter, mesh.coords)
	u0 = zeros(lifter.size_reduced)
	# a 20 % stretch in one step: the Hessian is indefinite on the way (modified Newton steps)
	reference = newton(p, u0)
	@test reference.converged
	r = newton(p, u0; linear_solver = LinearSolveSolver(KLUFactorization()))
	@test r.converged && r.u ≈ reference.u
	# conjugate gradients (positive definite Hessians: a small step) with smoothed-aggregation
	# multigrid, the rigid body modes as near null space
	p_small = with_values(p, :stretch => 0.02)
	reference = newton(p_small, u0)
	B = rigid_body_modes(mesh.coords; lifter)
	amg(K) = aspreconditioner(smoothed_aggregation(K; B))
	solver = LinearSolveSolver(KrylovJL_CG(); preconditioner = amg, reltol = 1e-12)
	r = newton(p_small, u0; linear_solver = solver)
	@test r.converged && r.u ≈ reference.u
	@test solver.cache !== nothing  # kept for the next solve
end
