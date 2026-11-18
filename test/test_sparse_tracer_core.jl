# Port of tatva/tests/test_sparse_tracer_core.py.
#
# tatva's tracer interprets JAX primitives; the Julia port traces by operator overloading
# (SparseConnectivityTracer), so primitive-level cases become ordinary Julia code. The core
# contract is unchanged:
#   * no false negatives — the traced pattern contains the true (dense AD) pattern;
#   * an energy Hessian pattern is structurally symmetric;
#   * structural operations keep locality (no spurious global coupling).
# Tests of tatva's internal data structures (SparseDepSet, CouplingAccumulator,
# _unwrap_jit) have no counterpart and are not ported. tatva's documented soundness hole
# (a data-dependent scan carry, `xfail`) is traced exactly here and therefore tested as
# passing.

using Test, Tatva, SparseArrays, LinearAlgebra, ForwardDiff, Random

nz_set(m) = (A = sparse(m); Set{Tuple{Int, Int}}(zip(findnz(A)[1][nonzeros(A).!=0], findnz(A)[2][nonzeros(A).!=0])))

function dense_hessian_pattern(f, n; n_samples = 6, seed = 0)
	rng = MersenneTwister(seed)
	pattern = Set{Tuple{Int, Int}}()
	for _ ∈ 1:n_samples
		H = ForwardDiff.hessian(f, randn(rng, n))
		union!(pattern, Tuple.(findall(abs.(H) .> 1e-9)))
	end
	return pattern
end

function dense_vw_pattern(g, n; n_samples = 6, seed = 0)
	R(u) = ForwardDiff.gradient(w -> g(w, u), zeros(eltype(u), n))
	rng = MersenneTwister(seed)
	pattern = Set{Tuple{Int, Int}}()
	for _ ∈ 1:n_samples
		K = ForwardDiff.jacobian(R, randn(rng, n))
		union!(pattern, Tuple.(findall(abs.(K) .> 1e-9)))
	end
	return pattern
end

#--------------------------------------------------------------------------------
# A. property battery: one minimal energy per family
#--------------------------------------------------------------------------------

const _A = (B = randn(MersenneTwister(1), 6, 6); B + B')
const _PERM = [3, 1, 2, 2, 4, 6]
const _LO = [1, 2, 3]
const _HI = [4, 5, 6]
const _SCATTER_IDX = [1, 2, 3, 1, 2, 3]

function scatter_add(u)
	z = zeros(eltype(u), 3)
	for (k, i) ∈ enumerate(_SCATTER_IDX)
		z[i] += u[k]
	end
	return z
end
adjacent_product(u) = sum(u[1:(end-1)] .* u[2:end])  # a function barrier (tatva: nested jit)
function scan_squares(u)  # a scan without carry coupling
	ys = similar(u)
	for (i, x) ∈ enumerate(u)
		ys[i] = x^2
	end
	return sum(ys)
end

# (id, energy, n_dofs, exact)
const ENERGY_CASES = [
	("unary_sin_diagonal", u -> sum(sin.(u)), 6, true),
	("unary_dense_via_reduce", u -> sin(sum(u)), 6, true),
	("unary_exp", u -> sum(exp.(u)), 6, true),
	("unary_log_safe", u -> sum(log.(u .^ 2 .+ 1.0)), 6, false),
	("unary_tanh", u -> sum(tanh.(u)), 6, true),
	("binary_mul_adjacent", u -> sum(u[1:(end-1)] .* u[2:end]), 6, false),
	("binary_div_adjacent", u -> sum(u[1:(end-1)] ./ (u[2:end] .^ 2 .+ 1.0)), 6, false),
	("integer_pow2_diagonal", u -> sum(u .^ 2), 6, true),
	("integer_pow3_diagonal", u -> sum(u .^ 3), 6, true),
	("dot_quadratic_form", u -> u' * _A * u, 6, true),
	("reduce_sum_then_square", u -> sum(vec(sum(reshape(u, 3, 2); dims = 2)) .^ 2), 6, true),
	("reshape_transpose", u -> sum(sin.(vec(permutedims(reshape(u, 3, 2))))), 6, true),
	("pad", u -> sum(sin.(vcat(zero(eltype(u)), u, zero(eltype(u))))), 6, true),
	("broadcast", u -> sum(sin.(repeat(u, 1, 3))), 6, true),
	("concatenate_duplicate", u -> sum(sin.(vcat(u, u))), 6, true),
	("squeeze", u -> sum(dropdims(sin.(reshape(u, :, 1)); dims = 2)), 6, true),
	("slice_product", u -> sum(u[1:3] .* u[4:6]), 6, false),
	("gather_permutation", u -> sum(sin.(u[_PERM])), 6, true),
	("gather_product", u -> sum(u[_LO] .* u[_HI]), 6, false),
	("scatter_add_then_square", u -> sum(scatter_add(u) .^ 2), 6, true),
	("where_two_branches", u -> sum(ifelse.(u .> 0, u .^ 2, u .^ 3)), 6, true),
	("slice_squared", u -> sum(u[2:4] .^ 2), 6, true),  # tatva: dynamic_slice, conservative there
	("function_barrier", adjacent_product, 6, false),
	("map_independent", u -> sum(map(x -> x^2, u)), 6, true),
	("map_batched", u -> sum(mapreduce(c -> c .^ 2, vcat, Iterators.partition(u, 2))), 6, true),
	("scan_map_like", scan_squares, 6, true),
]

@testset "energy: no false negatives: $name" for (name, f, n, _) in ENERGY_CASES
	@test dense_hessian_pattern(f, n) ⊆ nz_set(pattern_from_energy(f, n))
end

@testset "energy: Hessian pattern is symmetric: $name" for (name, f, n, _) in ENERGY_CASES
	pat = pattern_from_energy(f, n)
	@test nz_set(pat) == nz_set(sparse(transpose(pat)))
end

@testset "energy: exact pattern: $name" for (name, f, n, exact) in ENERGY_CASES
	exact || continue
	@test nz_set(pattern_from_energy(f, n)) == dense_hessian_pattern(f, n)
end

@testset "zero Hessian returns the identity: $name" for (name, f) in (
	("linear", u -> 3.0 * sum(u)),
	("constant", u -> 5.0 + 0.0 * u[1]),
	("floor", u -> sum(floor.(u))),
	("comparison", u -> sum(u .> 0.0)),
)
	@test nz_set(pattern_from_energy(f, 4)) == nz_set(sparse(I, 4, 4))
end

#--------------------------------------------------------------------------------
# B. nonlinear classification
#--------------------------------------------------------------------------------

@testset "integer power exponent classification: $p" for (p, records) in ((-1, true), (0, false), (1, false), (2, true), (3, true))
	f = u -> sum((u .+ 2.0) .^ p)
	pat = pattern_from_energy(f, 4)
	# records: genuine diagonal curvature; otherwise the identity fallback (the same set here)
	@test nz_set(pat) == Set((i, i) for i in 1:4)
	records && @test dense_hessian_pattern(f, 4) ⊆ nz_set(pat)
end

@testset "linear scaling is not recorded: $name" for (name, f) in (
	("const_mul_lhs", u -> sum(3.0 .* u)),
	("const_mul_rhs", u -> sum(u .* 2.0)),
	("const_div", u -> sum(u ./ 4.0)),
)
	@test nz_set(pattern_from_energy(f, 5)) == nz_set(sparse(I, 5, 5))
end

const _C5 = collect(1.0:5.0)
const _M5 = reshape(collect(1.0:25.0), 5, 5)

@testset "contraction with a constant operand is not recorded: $name" for (name, f) in (
	("dot_const_lhs", u -> dot(_C5, u)),
	("dot_const_rhs", u -> dot(u, _C5)),
	("const_matvec", u -> sum(_M5 * u)),
)
	@test nz_set(pattern_from_energy(f, 5)) == nz_set(sparse(I, 5, 5))
end

@testset "a bilinear contraction still couples" begin
	f = u -> u' * _M5 * u
	@test dense_hessian_pattern(f, 5) ⊆ nz_set(pattern_from_energy(f, 5))
end

cross_segment_pairs(pairs, block) = Set(p for p in pairs if (p[1] - 1) ÷ block != (p[2] - 1) ÷ block)

@testset "batched matvec with a constant normal keeps per-segment locality (d=$d)" for d in (2, 3)
	N = 6
	b = d * d
	n = N * b
	normal = collect(1.0:d)
	function energy(u)
		S = reshape(u, d, d, N)  # segment k owns the contiguous dofs (k-1)b+1 : kb
		e = zero(eltype(u))
		for k ∈ 1:N
			Sk = (S[:, :, k] + transpose(S[:, :, k])) / 2
			traction = Sk * normal
			jump = Sk[1, :]
			e += dot(traction, jump)
		end
		return e
	end
	pat = pattern_from_energy(energy, n)
	traced = nz_set(pat)
	@test dense_hessian_pattern(energy, n) ⊆ traced
	@test isempty(cross_segment_pairs(traced, b))
	@test nnz(pat) <= N * b * b

	# the same written per segment and mapped (tatva: autovmap), normal as a closure constant
	density(Sk) = (Ss = (Sk + transpose(Sk)) / 2; dot(Ss * normal, Ss[1, :]))
	energy2(u) = sum(density(reshape(u, d, d, N)[:, :, k]) for k in 1:N)
	pat2 = pattern_from_energy(energy2, n)
	@test dense_hessian_pattern(energy2, n) ⊆ nz_set(pat2)
	@test isempty(cross_segment_pairs(nz_set(pat2), b))
	@test nnz(pat2) <= N * b * b
end

@testset "dense linear algebra couples even when consumed linearly: $op" for op in (:inv, :solve)
	n = 9
	rhs = collect(1.0:3.0)
	f = op === :inv ? (u -> sum(inv(reshape(u, 3, 3) + 5.0I))) : (u -> sum((reshape(u, 3, 3) + 5.0I) \ rhs))
	@test dense_hessian_pattern(f, n) ⊆ nz_set(pattern_from_energy(f, n))
end

@testset "stacking keeps per-slice support" begin
	N, block = 4, 2
	energy(u) = sum([sin(u[2i-1]) * u[2i] for i in 1:N] .^ 2)
	traced = nz_set(pattern_from_energy(energy, block * N))
	@test dense_hessian_pattern(energy, block * N) ⊆ traced
	@test isempty(cross_segment_pairs(traced, block))
end

@testset "splitting keeps per-piece support" begin
	N, block = 4, 2
	n = block * N
	energy(u) = sum(sin(p[1]) * p[2] for p in Iterators.partition(u, block))
	pat = pattern_from_energy(energy, n)
	traced = nz_set(pat)
	@test dense_hessian_pattern(energy, n) ⊆ traced
	@test isempty(cross_segment_pairs(traced, block))
	@test nnz(pat) <= N * block * block

	# every piece is traced (the last one alone couples dofs 5 and 6)
	last_piece(u) = (c = u[5:6]; c[1] * c[2])
	t2 = nz_set(pattern_from_energy(last_piece, 6))
	@test Set([(5, 6), (6, 5)]) ⊆ t2
	@test isempty(cross_segment_pairs(t2, 2))
end

@testset "views and copies trace identically (axis $axis)" for axis in (1, 2)
	n = 8
	pieces(m) = axis == 1 ? (m[1:2, :], m[3:4, :]) : (m[:, 1:1], m[:, 2:2])
	vpieces(m) = axis == 1 ? (view(m, 1:2, :), view(m, 3:4, :)) : (view(m, :, 1:1), view(m, :, 2:2))
	with_copies(u) = ((a, b) = pieces(reshape(u, 4, 2)); sum(a .* b) + sum(sin.(a)))
	with_views(u) = ((a, b) = vpieces(reshape(u, 4, 2)); sum(a .* b) + sum(sin.(a)))
	pat = pattern_from_energy(with_views, n)
	@test nz_set(pat) == nz_set(pattern_from_energy(with_copies, n))
	@test dense_hessian_pattern(with_views, n) ⊆ nz_set(pat)
	@test nz_set(pat) == nz_set(sparse(transpose(pat)))
	@test nnz(pat) < n * n
end

@testset "tracing emits no warnings and is exact for structural code" begin
	energy(u) = sum(sum(p .^ 2) for p in (u[1:2], u[3:4]))
	pat = @test_logs pattern_from_energy(energy, 4)
	@test nz_set(pat) == Set((i, i) for i in 1:4)
	handled(u) = (A = reshape(u, 3, 3) + 5.0I; sum(inv(A)) + sum([sin(u[1]) * u[2]] .^ 2))
	@test_logs pattern_from_energy(handled, 9)
end

@testset "opaque functions are dense nonlinear blocks (eigenvalues)" begin
	for consume ∈ (w -> sum(w .^ 2), w -> sum(w))
		f = u -> consume(opaque(A -> eigvals(Symmetric(A)), reshape(u, 3, 3) + transpose(reshape(u, 3, 3))))
		@test dense_hessian_pattern(f, 9) ⊆ nz_set(pattern_from_energy(f, 9))
	end
end

@testset "reverse is a structural permutation" begin
	n = 6
	f = u -> sum(reverse(u) .* u)
	traced = nz_set(pattern_from_energy(f, n))
	@test dense_hessian_pattern(f, n) ⊆ traced
	@test all(c in (r, n + 1 - r) for (r, c) in traced)
end

@testset "non-generic code is reported, and works through `opaque`" begin
	# tatva warns and over-approximates an unhandled primitive; a global tracer cannot sort
	@test_throws TypeError pattern_from_energy(u -> sum(sort(u) .^ 2), 6)
	f = u -> sum(opaque(sort, u) .^ 2)
	@test dense_hessian_pattern(f, 6) ⊆ nz_set(pattern_from_energy(f, 6))
end

@testset "unary nonlinearity is diagonal, products couple operands" begin
	@test nz_set(pattern_from_energy(u -> sum(sin.(u)), 5)) == Set((i, i) for i in 1:5)
	n = 5
	adjacent = Set([(i, i + 1) for i in 1:(n-1)]) ∪ Set([(i + 1, i) for i in 1:(n-1)])
	@test adjacent ⊆ nz_set(pattern_from_energy(u -> sum(u[1:(end-1)] .* u[2:end]), n))
end

#--------------------------------------------------------------------------------
# D. control flow
#--------------------------------------------------------------------------------

@testset "function barriers do not change the pattern" begin
	e = u -> sum(u[1:(end-1)] .* u[2:end]) + sum(sin.(u))
	wrapped = u -> (inner(v) = e(v); inner(u))
	@test nz_set(pattern_from_energy(wrapped, 6)) == nz_set(pattern_from_energy(e, 6))
end

@testset "branch: a nonlinearity inside a branch is captured" begin
	n = 6
	f = u -> branch(u[1] > 0, v -> sum(v[1:(end-1)] .* v[2:end]), v -> sum(v .^ 2), u)
	pat = pattern_from_energy(f, n)
	@test dense_hessian_pattern(f, n) ⊆ nz_set(pat)
	@test nz_set(pat) == dense_hessian_pattern(f, n)  # tridiagonal ∪ diagonal
	@test f(ones(n)) == 5.0 && f(-ones(n)) == 6.0      # plain evaluation takes one branch
end

@testset "branch: a nonlinearity after affine branches is sound" begin
	n = 6
	f = u -> sum(branch(u[1] > 0, v -> v .* 2.0, v -> reverse(v) .* 3.0, u) .^ 2)
	@test dense_hessian_pattern(f, n) ⊆ nz_set(pattern_from_energy(f, n))
end

@testset "switch: every branch is traced" begin
	n = 6
	branches = (v -> sum(v .^ 2), v -> sum(v[1:(end-1)] .* v[2:end]), v -> sum(sin.(v)))
	f = u -> switch(clamp(u[1], 1, 3), branches, u)
	@test dense_hessian_pattern(f, n) ⊆ nz_set(pattern_from_energy(f, n))
	@test switch(2, branches, ones(3)) == 2.0
	@test switch(7, branches, zeros(3)) == 0.0  # clamped to the last branch
end

@testset "a loop carry coupling is traced exactly (tatva: xfail for lax.scan)" begin
	n = 6
	function f(u)  # ys = [1⋅x₁, x₁x₂, x₂x₃, ...]
		c = one(eltype(u))
		s = zero(eltype(u))
		for x ∈ u
			s += c * x
			c = x
		end
		return s
	end
	@test dense_hessian_pattern(f, n) ⊆ nz_set(pattern_from_energy(f, n))
	@test nz_set(pattern_from_energy(f, n)) == dense_hessian_pattern(f, n)
end

#--------------------------------------------------------------------------------
# E. virtual-work trial/test split
#--------------------------------------------------------------------------------

vw_kappa_default(w, u, kappa = 3.0) = sum(kappa .* w[1:(end-1)] .* u[2:end])
vw_kappa_required(w, u, kappa) = sum(kappa .* w .* u)

@testset "virtual work: a bad argument name raises" begin
	g = (w, u) -> sum(w .* u)
	@test_throws "Trial argument 'x' not found" pattern_from_virtual_work(g, 3, :x, :w)
	@test_throws "Test argument 'y' not found" pattern_from_virtual_work(g, 3, "u", "y")
end

@testset "virtual work: a trial-only term does not leak into K" begin
	n = 4
	g = (w, u) -> sum(w .* u) + sum(u .^ 2)
	@test nz_set(pattern_from_virtual_work(g, n, :u, :w)) == Set((i, i) for i in 1:n)
end

@testset "virtual work: static arguments are forwarded" begin
	n = 4
	g = (w, u, kappa) -> sum(kappa .* w[1:(end-1)] .* u[2:end])
	K = pattern_from_virtual_work(g, n, :u, :w, 2.0)
	@test dense_vw_pattern((w, u) -> g(w, u, 2.0), n) ⊆ nz_set(K)
	@test nnz(K) > 0
	# positions instead of names
	@test pattern_from_virtual_work(g, n, 2, 1, 2.0) == K
end

@testset "virtual work: default arguments are used" begin
	n = 4
	K = pattern_from_virtual_work(vw_kappa_default, n, :u, :w)
	@test dense_vw_pattern((w, u) -> vw_kappa_default(w, u), n) ⊆ nz_set(K)
end

@testset "virtual work: a missing static argument raises" begin
	@test_throws "Missing static argument" pattern_from_virtual_work(vw_kappa_required, 3, :u, :w)
end

@testset "virtual work: an unsymmetric pattern is captured" begin
	n = 4
	g = (w, u) -> sum(w[1:(end-1)] .* u[2:end])  # w_i couples to u_{i+1} only
	K = pattern_from_virtual_work(g, n, :u, :w)
	@test dense_vw_pattern(g, n) ⊆ nz_set(K)
	@test nz_set(K) != Set((c, r) for (r, c) in nz_set(K))
end

#--------------------------------------------------------------------------------
# G. opaque leaves (tatva: custom_jvp / custom_vjp / callbacks / debug / ffi)
#--------------------------------------------------------------------------------

adjacent_energy(leaf, M) = U -> sum(leaf(U[i] - U[i+1]) for i in 1:(M-1))
tridiagonal(M) = Set((i, j) for i in 1:M, j in 1:M if abs(i - j) <= 1)

@testset "a smooth leaf gives the exact tridiagonal pattern" begin
	M = 6
	fn = adjacent_energy(x -> 0.5 * x^2, M)
	@test nz_set(pattern_from_energy(fn, M)) == tridiagonal(M)
	@test dense_hessian_pattern(fn, M) ⊆ nz_set(pattern_from_energy(fn, M))
end

@testset "a leaf hiding an iterative solve is traced through the iterations" begin
	M = 6
	function leaf(x)
		y = x
		for _ ∈ 1:5  # Newton iterations for y³ + y = x
			y = y - (y^3 + y - x) / (3y^2 + 1)
		end
		return 0.5 * y^2
	end
	@test nz_set(pattern_from_energy(adjacent_energy(leaf, M), M)) == tridiagonal(M)
end

@testset "an opaque (non-traceable) leaf couples its inputs" begin
	M = 6
	external(x) = 0.5 * x^2  # stands for a call into a foreign library
	pat = pattern_from_energy(adjacent_energy(x -> opaque(external, x), M), M)
	@test nz_set(pat) == tridiagonal(M)
	@test nnz(pat) > M
	@test opaque(external, 3.0) == 4.5  # plain numbers just call the function
end

@testset "printing inside an energy does not change the pattern" begin
	M = 6
	base = U -> sum(U[1:(end-1)] .* U[2:end])
	with_print = U -> (print(devnull, "tracing ", size(U)); base(U))
	@test nz_set(pattern_from_energy(with_print, M)) == nz_set(pattern_from_energy(base, M))
end

@testset "an opaque call per quadrature point keeps the pattern sparse (tatva: ffi)" begin
	M = 6
	leaf(s) = 2.0 .* s  # a per-point external constitutive update
	function energy_batched(U)
		strain = reshape(U[1:(end-1)] - U[2:end], 1, :)
		return sum(opaque(leaf, strain) .* strain)  # one opaque call for all points
	end
	function energy_pointwise(U)
		strain = reshape(U[1:(end-1)] - U[2:end], 1, :)
		return sum(sum(opaque(leaf, s) .* s) for s in eachcol(strain))
	end
	@test nnz(pattern_from_energy(energy_batched, M)) == M * M  # conservative: dense
	@test nz_set(pattern_from_energy(energy_pointwise, M)) == tridiagonal(M)
end
