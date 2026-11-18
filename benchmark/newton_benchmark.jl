# Nonlinear solves of the 3D neo-Hookean bar of examples/hyperelasticity.jl (5 load steps to
# 25 % elongation), timing the whole solve:
#
#   reduce:  assemble the full gradient and Hessian, reduce them with the lifter, solve with a
#            fresh sparse LU every iteration (the loop of the examples before `newton`)
#   newton:  `EnergyProblem` (assembly straight into the free dofs) + `newton` with the
#            LU (`DirectSolver`) or Cholesky (`CholeskySolver`) factorization kept across
#            iterations and load steps
#
#     julia --project=. -t auto benchmark/newton_benchmark.jl

using Tatva, LinearAlgebra, SparseArrays, Printf

const μ, λ = 1.0, 1.5
ψ(∇u) = (F = I + ∇u; J = det(F); μ / 2 * (tr(F' * F) - 3) - μ * log(J) + λ / 2 * log(J)^2)

include(joinpath(@__DIR__, "..", "examples", "box_mesh.jl"))

function problem(n; threaded)
	mesh = box_mesh(4n, n, n)
	op = Operator(mesh, Tetrahedron4(); cache_weights = true)
	x = mesh.coords[1, :]
	left, right = findall(x .== 0.0), findall(x .== 4.0)
	d(c, nodes) = 3 .* (nodes .- 1) .+ c
	lifter = Lifter(3 * num_nodes(mesh), Fixed(vcat(d(1, left), d(2, left), d(3, left)), 0.0),
		Fixed(d(1, right), RuntimeValue(:stretch, 0.0)), Fixed(vcat(d(2, right), d(3, right)), 0.0))
	asm = ElementAssembler(op, 3; threaded)
	density(ξ, Ue, Xe) = ψ(grad(op.element, ξ, Ue, Xe))
	return asm, density, lifter, mesh
end

const STRETCHES = range(0.0, 1.0; length = 6)[2:end]

function solve_reduce(asm, density, lifter, mesh)
	u = zeros(lifter.size_reduced)
	g, K = zeros(lifter.size), similar(asm.pattern)
	for stretch ∈ STRETCHES
		lf = with_values(lifter, :stretch => stretch)
		for _ ∈ 1:15
			assemble!(g, K, asm, density, reshape(lift_from_zeros(lf, u), 3, :), mesh.coords)
			r = reduce_adjoint(lf, g)
			norm(r) < 1e-9 && break
			u .-= spsolve(reduce_hessian(lf, K), r)
		end
	end
	return u
end

function solve_newton(asm, density, lifter, mesh, solver)
	p = EnergyProblem(asm, density, lifter, mesh.coords)
	u = zeros(lifter.size_reduced)
	for stretch ∈ STRETCHES
		r = newton(with_values(p, :stretch => stretch), u; linear_solver = solver, atol = 1e-9, rtol = 0.0)
		r.converged || error("no convergence at stretch $stretch")
		u = r.u
	end
	return u
end

println("Julia ", VERSION, ", ", Threads.nthreads(), " threads")
for n ∈ (8, 12)
	for threaded ∈ unique((false, Threads.nthreads() > 1))
		asm, density, lifter, mesh = problem(n; threaded)
		solve_newton(asm, density, lifter, mesh, CholeskySolver())  # compile
		t_lu = @elapsed u2 = solve_newton(asm, density, lifter, mesh, DirectSolver())
		t_chol = @elapsed u3 = solve_newton(asm, density, lifter, mesh, CholeskySolver())
		@assert u2 ≈ u3
		# plain Newton steps may turn elements inside out (no line search)
		t_reduce = try
			t = @elapsed u1 = solve_reduce(asm, density, lifter, mesh)
			@assert u1 ≈ u3
			t
		catch err
			Tatva._is_domain_error(err) || rethrow()
			nothing
		end
		@printf("%6d free dofs, %s: reduce + LU %s | newton, LU %6.2f s | newton, Cholesky %6.2f s\n",
			lifter.size_reduced, threaded ? "threaded" : "1 thread",
			t_reduce === nothing ? "  fails" : @sprintf("%6.2f s", t_reduce), t_lu, t_chol)
	end
end
