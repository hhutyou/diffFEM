# Port of tatva/tests/test_lifter.py and tests/test_lifter_residual.py, plus property tests
# of the lifter (adjointness, sparsity adaptation, runtime values).

using Test, Tatva, SparseArrays, LinearAlgebra, ForwardDiff, Random

@testset "lifter without constraints round-trips" begin
	lifter = Lifter(4)
	u_reduced = collect(0.0:(lifter.size_reduced-1))
	u_full = lift_from_zeros(lifter, u_reduced)
	@test u_full == [0.0, 1.0, 2.0, 3.0]
	@test reduce(lifter, u_full) == u_reduced
	@test lifter.constrained_dofs == Int[]
end

@testset "lifter applies Dirichlet and periodic constraints" begin
	lifter = Lifter(6, Fixed([1, 6]), Periodic(dofs = [3], master_dofs = [2]))
	u_reduced = [10.0, 20.0, 30.0]
	lifted_u = lift_from_zeros(lifter, u_reduced)
	@test lifted_u == [0.0, 10.0, 10.0, 20.0, 30.0, 0.0]
	@test reduce(lifter, lifted_u) == u_reduced
	# lifting on top of a previous full vector keeps nothing but the constraints/free dofs
	@test lift(lifter, u_reduced, fill(7.0, 6)) == lifted_u
end

@testset "lifter deduplicates overlapping constraints" begin
	lifter = Lifter(7, Fixed([1, 3]), Periodic(dofs = [3, 7], master_dofs = [2, 6]))
	@test lifter.constrained_dofs == [1, 3, 7]
	@test lifter.free_dofs == [2, 4, 5, 6]
	@test lifter.size_reduced == 4
end

@testset "constraints and lifter are hashable" begin
	periodic = Periodic(dofs = [3], master_dofs = [2])
	dirichlet = Fixed([1, 6], [0.0, 1.0])
	lifter = Lifter(6, dirichlet, periodic)
	@test hash(periodic) isa UInt
	@test hash(dirichlet) isa UInt
	@test hash(lifter) isa UInt
	@test periodic == periodic
	@test periodic != Periodic(dofs = [3], master_dofs = [2])  # identity-based equality, as in tatva
	@test Dict(lifter => 1)[lifter] == 1
end

# tatva: "lifter as a static argument in jit". The Julia counterpart: a lifter captured in
# a closure that is differentiated — lifting is affine, its Jacobian selects the free dofs.
@testset "lifter inside differentiated code" begin
	lifter = Lifter(6, Fixed([1, 6], 2.0), Periodic(dofs = [3], master_dofs = [2]))
	u_reduced = [10.0, 20.0, 30.0]
	@test lift_from_zeros(lifter, u_reduced) == [2.0, 10.0, 10.0, 20.0, 30.0, 2.0]
	J = ForwardDiff.jacobian(u -> lift_from_zeros(lifter, u), u_reduced)
	@test J == [0 0 0; 1 0 0; 1 0 0; 0 1 0; 0 0 1; 0 0 0]
	# differentiating with respect to a Dirichlet value
	ddisp = ForwardDiff.derivative(d -> sum(lift_from_zeros(Lifter(4, Fixed([1, 4], d)), [1.0, 1.0])), 0.3)
	@test ddisp ≈ 2.0
end

@testset "lifter equality handles array runtime values" begin
	lifter = Lifter(4, Fixed([1, 4], RuntimeValue("top")))
	lhs = with_values(lifter, Dict("top" => [1.0, 2.0]))
	rhs = with_values(lifter, "top" => [1.0, 2.0])
	diff = with_values(lifter, "top" => [1.0, 3.0])
	@test lhs == rhs
	@test lhs != diff
	@test lift_from_zeros(lhs, [5.0, 6.0]) == [1.0, 5.0, 6.0, 2.0]
end

@testset "runtime values: defaults, `at[...].set`, errors" begin
	lifter = Lifter(4, Fixed([1], RuntimeValue("bottom", 0.0)), Fixed([4], RuntimeValue("top", 0.5)))
	@test lift_from_zeros(lifter, [1.0, 2.0]) == [0.0, 1.0, 2.0, 0.5]
	moved = lifter.at["top"].set(1.5)
	@test lift_from_zeros(moved, [1.0, 2.0]) == [0.0, 1.0, 2.0, 1.5]
	@test lift_from_zeros(lifter, [1.0, 2.0])[4] == 0.5  # the original is unchanged
	@test_throws LifterError with_values(lifter, "nope" => 1.0)
	missing_value = Lifter(3, Fixed([2], RuntimeValue(:load)))
	@test_throws LifterError lift_from_zeros(missing_value, [1.0, 2.0])
	@test lift_from_zeros(with_values(missing_value, :load => 9.0), [1.0, 2.0]) == [1.0, 9.0, 2.0]
end

@testset "lifted as a function decorator" begin
	lifter = Lifter(4, Fixed([1, 4], 0.0))
	my_fn = lifted(u_full -> sum(u_full); argnums = 1)
	@test my_fn(lifter, [10.0, 20.0]) == 30.0
	reduce_out = lifted(u_full -> 2 .* u_full; argnums = 1, output = :primal)
	@test reduce_out(lifter, [10.0, 20.0]) == [20.0, 40.0]
	# a lifted function used inside another function (tatva: "lifted function is a pytree")
	step(u) = my_fn(lifter, u)
	@test step([10.0, 20.0]) == 30.0
	solve(u_reduced, l) = lifted(u -> sum(u); argnums = 1)(l, u_reduced)
	@test solve([10.0, 20.0], lifter) == 30.0
	@test_throws LifterError my_fn(lifter, [1.0, 2.0, 3.0])
	@test_throws LifterError lifted(sum; output = :nope)
end

#--------------------------------------------------------------------------------
# tests/test_lifter_residual.py
#--------------------------------------------------------------------------------

@testset "reduce_adjoint with a periodic constraint" begin
	lifter = Lifter(4, Periodic(dofs = [3], master_dofs = [2]))
	@test reduce_adjoint(lifter, [10.0, 20.0, 30.0, 40.0]) == [10.0, 50.0, 40.0]
end

@testset "reduce_adjoint with fixed dofs" begin
	lifter = Lifter(4, Fixed([1, 4], 0.0))
	@test reduce_adjoint(lifter, [10.0, 20.0, 30.0, 40.0]) == [20.0, 30.0]
end

@testset "lifted dual output" begin
	lifter = Lifter(4, Periodic(dofs = [3], master_dofs = [2]))
	compute_residual = lifted(u_full -> u_full .* 2.0; argnums = 1, output = :dual)
	# lift → [1, 2, 2, 3]; × 2 → [2, 4, 4, 6]; reduce_adjoint → [2, 4 + 4, 6]
	@test compute_residual(lifter, [1.0, 2.0, 3.0]) == [2.0, 8.0, 6.0]
end

@testset "reduce_adjoint with multiple constraints (reverse order)" begin
	lifter = Lifter(4, Periodic(dofs = [3], master_dofs = [2]), Fixed([4], 0.0))
	@test reduce_adjoint(lifter, [10.0, 20.0, 30.0, 40.0]) == [10.0, 50.0]
end

#--------------------------------------------------------------------------------
# properties
#--------------------------------------------------------------------------------

@testset "reduce_adjoint is the adjoint of the linear part of lift" begin
	rng = MersenneTwister(7)
	lifter = Lifter(10, Fixed([1, 5], 3.0), Periodic([2, 9], [4, 6]), Periodic([10], [2]))
	for _ ∈ 1:5
		δ = randn(rng, lifter.size_reduced)
		r = randn(rng, lifter.size)
		Lδ = lift_from_zeros(lifter, δ) - lift_from_zeros(lifter, zeros(lifter.size_reduced))
		@test dot(reduce_adjoint(lifter, r), δ) ≈ dot(r, Lδ)
	end
	# hence reduce_adjoint(∇E(lift(u))) is the gradient of the reduced energy
	E(u) = sum(abs2, u) + u[2] * u[9]^2
	Er(ur) = E(lift_from_zeros(lifter, ur))
	ur = randn(rng, lifter.size_reduced)
	@test reduce_adjoint(lifter, ForwardDiff.gradient(E, lift_from_zeros(lifter, ur))) ≈ ForwardDiff.gradient(Er, ur)
end

@testset "chained and overlapping periodic constraints: adjoint and sparsity" begin
	lifters = (
		Lifter(4, Periodic(2, 1), Periodic(3, 2)),            # a slave is the master of a later constraint
		Lifter(4, Periodic(3, 2), Periodic(2, 1)),            # ... of an earlier one
		Lifter(4, Periodic(2, 1), Periodic([2, 3], [1, 2])),  # a master is a slave of the same constraint
		Lifter(6, Periodic([5], [3]), Fixed([6], 1.0), Periodic([3, 4], [1, 5])),
	)
	for lifter ∈ lifters
		n = lifter.size
		P = lifting_matrix(lifter)
		v = collect(1.0:lifter.size_reduced)
		@test P * v ≈ lift_from_zeros(lifter, v) - lift_from_zeros(lifter, zero(v))
		r = collect(1.0:n) .^ 2
		@test reduce_adjoint(lifter, r) ≈ transpose(P) * r
		# the adapted pattern holds that of Pᵀ S P for a coupling between any two dofs
		B = Int.(Matrix(P) .!= 0)
		holds(i, j) = (S = sparse([i], [j], [true], n, n); all(Matrix(adapt_sparsity(lifter, S)) .>= (transpose(B) * Matrix(S) * B .> 0)))
		@test all(holds(i, j) for i in 1:n, j in 1:n)
	end
	@test_throws ArgumentError Periodic([2, 2], [1, 3])  # a dof with two masters
end

@testset "sparsity adaptation for periodic constraints" begin
	# 1D chain 1-2-3-4 with dof 4 periodic to dof 1: the reduced pattern closes the ring
	S = sparse([1, 2, 2, 3, 3, 4, 1, 2, 3, 4], [2, 1, 3, 2, 4, 3, 1, 2, 3, 4], fill(true, 10), 4, 4)
	lifter = Lifter(4, Periodic([4], [1]))
	aug = augment_sparsity(lifter, S)
	@test aug[1, 3] && aug[3, 1]  # master 1 inherits the coupling 4–3
	red = adapt_sparsity(lifter, S)
	@test size(red) == (3, 3)
	@test Matrix(red) == trues(3, 3)
	# the adapted pattern contains the pattern of the reduced Hessian
	E(u) = sum((u[i+1] - u[i])^2 * (1 + u[i]^2) for i in 1:3)
	Hr = ForwardDiff.hessian(ur -> E(lift_from_zeros(lifter, ur)), [0.1, 0.2, 0.3])
	@test all(red[i, j] for i in 1:3, j in 1:3 if abs(Hr[i, j]) > 1e-12)
	# patterns smaller than the lifter are padded
	@test size(augment_sparsity(Lifter(5), S)) == (5, 5)
	@test reduce_sparsity(Lifter(4, Fixed([2])), S) == S[[1, 3, 4], [1, 3, 4]]
end
