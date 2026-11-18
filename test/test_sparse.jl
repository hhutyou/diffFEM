# Port of tatva/tests/test_sparse.py, plus tests of the coloring, ColoredMatrix and the
# differentiable sparse solve.

using Test, Tatva, LinearAlgebra, SparseArrays, StaticArrays, ForwardDiff, Random

strain(∇u) = (∇u + ∇u') / 2
stress(ε, μ, λ) = 2μ * ε + λ * tr(ε) * I
strain_energy(∇u, μ, λ) = (ε = strain(∇u); 0.5 * dot(stress(ε, μ, λ), ε))

# no two columns of the same color share a row
function is_valid_column_coloring(A::SparseMatrixCSC, colors)
	At = sparse(transpose(A))
	for r ∈ 1:size(A, 1)
		cols = rowvals(At)[nzrange(At, r)]
		length(unique(colors[cols])) == length(cols) || return false
	end
	return true
end

@testset "sparse jacfwd of the gradient equals the dense Hessian" begin
	mesh = unit_square(8, 8)
	op = Operator(mesh, Tri3())
	total_energy(u_flat) = integrate(op, strain_energy.(grad(op, reshape(u_flat, 2, :)), 1.0, 0.0))
	n_dofs = 2 * num_nodes(mesh)
	u0 = zeros(n_dofs)
	K = ForwardDiff.hessian(total_energy, u0)

	pattern = pattern_from_mesh(mesh, 2)
	cm0 = ColoredMatrix(pattern)
	colors, seeds = distance2_color_and_seeds(cm0.indptr, cm0.indices, n_dofs)  # tatva's CSR signature
	@test size(seeds) == (maximum(colors), n_dofs)
	cm = ColoredMatrix(pattern; colors = colors)

	K_sparse = jacfwd(jacrev(total_energy), cm; color_batch_size = 10)(u0)
	@test K ≈ to_dense(K_sparse) atol = 1e-12
	primal, K_lin = linearized_jacfwd(jacrev(total_energy), cm; color_batch_size = 10)(u0)
	@test K ≈ to_dense(K_lin) atol = 1e-12
	@test primal ≈ ForwardDiff.gradient(total_energy, u0) atol = 1e-12

	# a nonlinear energy at a nonzero state, all colors in one evaluation
	ψ(∇u) = (F = I + ∇u; C = F' * F; J = det(F); 0.5 * (tr(C) - 2) - log(J) + 0.5 * log(J)^2)
	E(u) = integrate(op, ψ.(grad(op, reshape(u, 2, :))))
	u1 = 0.005 .* randn(MersenneTwister(1), n_dofs)  # small enough that det F > 0 everywhere
	@test to_dense(sparse_hessian(E, cm)(u1)) ≈ ForwardDiff.hessian(E, u1) atol = 1e-10
end

@testset "jacfwd with extra arguments and keywords" begin
	mesh = unit_square(4, 4)
	op = Operator(mesh, Tri3())
	E(u, damage; scale = 1.0) = scale * integrate(op, strain_energy.(grad(op, reshape(u, 2, :)), 1.0, 0.5)) * sum(damage)
	cm = ColoredMatrix(pattern_from_mesh(mesh, 2))
	n = 2 * num_nodes(mesh)
	d = fill(0.5, num_nodes(mesh))
	K = jacfwd(jacrev(E), cm)(zeros(n), d; scale = 2.0)
	@test to_dense(K) ≈ ForwardDiff.hessian(u -> E(u, d; scale = 2.0), zeros(n)) atol = 1e-12
end

@testset "jacfwd of an unsymmetric residual" begin
	# R_i = u_i^2 + u_{i+1}: lower-bidiagonal-free, unsymmetric Jacobian
	R(u) = [i < length(u) ? u[i]^2 + 3u[i+1] : u[i]^2 for i in eachindex(u)]
	n = 7
	pattern = sparse([1:n; 1:(n-1)], [1:n; 2:n], fill(true, 2n - 1), n, n)
	cm = ColoredMatrix(pattern)
	@test is_valid_column_coloring(pattern, cm.colors)
	u = collect(1.0:n)
	@test to_dense(jacfwd(R, cm)(u)) ≈ ForwardDiff.jacobian(R, u)
	@test to_dense(jacfwd(R, cm; color_batch_size = 1)(u)) ≈ ForwardDiff.jacobian(R, u)
end

@testset "sparse_hessian: symmetric colorings need fewer colors" begin
	mesh = unit_square(6, 5)
	op = Operator(mesh, Tri3())
	ψ(∇u) = (F = I + ∇u; J = det(F); 0.5 * (tr(F' * F) - 2) - log(J) + 0.6 * log(J)^2)
	E(u) = integrate(op, ψ.(grad(op, reshape(u, 2, :))))
	u = 0.01 .* sin.(1:2num_nodes(mesh))
	cm = ColoredMatrix(pattern_from_mesh(mesh, 2))
	H = ForwardDiff.hessian(E, u)
	for coloring ∈ (:acyclic, :star, :column), batch ∈ (nothing, 3)
		K = sparse_hessian(E, cm; coloring, color_batch_size = batch)(u)
		@test K.indptr == cm.indptr && K.indices == cm.indices
		@test to_dense(K) ≈ H atol = 1e-12
	end
	@test n_colors(sparse_hessian(E, cm)) < n_colors(cm) ÷ 2  # acyclic (substitution): 8 colors, distance-2: 18
	# dual numbers of an enclosing derivative through the decompression (weighted entries: the
	# plain sum of a translation-invariant Hessian is zero)
	E2(u, k) = k * E(u)
	w = collect(1.0:nnz(cm))
	@test ForwardDiff.derivative(k -> dot(w, sparse_hessian(E2, cm)(u, k).data), 0.5) ≈ dot(w, sparse_hessian(E, cm)(u).data)
	@test_throws ArgumentError sparse_hessian(E, cm; coloring = :bogus)
	@test_throws ArgumentError sparse_hessian(v -> sum(abs2, v), ColoredMatrix(sparse([1, 2, 2], [1, 1, 2], fill(true, 3), 2, 2)))
end

@testset "distance-2 colorings are valid: $name" for (name, colorfn) in (
	("greedy", distance2_colors),
	("largest degree first", largest_degree_first_distance2_colors),
	("smallest last", smallest_last_distance2_colors),
)
	for pattern ∈ (pattern_from_mesh(unit_square(6, 5), 2), pattern_from_mesh(unit_square(4, 4; type = :quad), 1),
		sprand(MersenneTwister(3), Bool, 40, 40, 0.08) .| sparse(I, 40, 40))
		colors = colorfn(pattern)
		@test is_valid_column_coloring(pattern, colors)
		@test minimum(colors) == 1
		cm = ColoredMatrix(pattern)
		@test colorfn(cm.indptr, cm.indices, size(pattern, 1)) == colors  # CSR signature
	end
	# a scalar Tri3 mesh needs few colors (one per node in the 2-ring)
	@test maximum(colorfn(pattern_from_mesh(unit_square(10, 10), 1))) <= 13
end

@testset "seeds_from_coloring" begin
	colors = [1, 2, 1, 3]
	@test seeds_from_coloring(colors) == [1 0 1 0; 0 1 0 0; 0 0 0 1]
end

@testset "ColoredMatrix conversions" begin
	A = sprand(MersenneTwister(5), 9, 7, 0.3)
	cm = ColoredMatrix(A)
	@test size(cm) == (9, 7)
	@test nnz(cm) == nnz(A)
	@test sparse(cm) == A
	@test to_dense(cm) == Matrix(A)
	@test Matrix(cm) == Matrix(A)
	# CSR layout: row pointers and column indices in row-major order
	rows, cols, vals = Tatva.to_coo(cm)
	@test issorted(rows)
	@test all(A[r, c] == v for (r, c, v) in zip(rows, cols, vals))
	@test cm.indptr[end] == nnz(A) + 1
	x = rand(7)
	@test cm * x ≈ A * x
	cm2 = with_data(cm, 2 .* cm.data)
	@test to_dense(cm2) == 2 .* Matrix(A)
	@test cm2.colors === cm.colors
	cm3 = ColoredMatrix(cm.indptr, cm.indices, cm.data, size(cm))
	@test cm3 == ColoredMatrix(A)
	@test_throws DimensionMismatch jacfwd(identity, ColoredMatrix(sparse(I, 3, 3)))(zeros(4))
end

@testset "spsolve, also through dual numbers" begin
	mesh = unit_square(5, 5)
	op = Operator(mesh, Tri3())
	n = num_nodes(mesh)
	cm = ColoredMatrix(pattern_from_mesh(mesh, 1))
	# mass-like matrix from AD: K = d/du ∫ (1 + k) u v + ∇u·∇v
	function residual(u, k)
		vw(v) = integrate(op, (1 + k) .* evaluate(op, v) .* evaluate(op, u)) + integrate(op, dot.(grad(op, v), grad(op, u)))
		return jacrev(vw)(zeros(eltype(u), n))
	end
	K(k) = jacfwd(residual, cm)(zeros(n), k)
	b = rand(MersenneTwister(9), n)
	@test spsolve(K(0.5), b) ≈ to_dense(K(0.5)) \ b
	B = rand(MersenneTwister(10), n, 3)
	@test spsolve(K(0.5), B) ≈ to_dense(K(0.5)) \ B
	# d/dk of the solution of K(k) x = b
	solve_k(k) = spsolve(K(k), b)
	dx = ForwardDiff.derivative(solve_k, 0.5)
	dx_dense = ForwardDiff.derivative(k -> to_dense(K(k)) \ b, 0.5)
	@test dx ≈ dx_dense atol = 1e-10
	# second derivatives (nested duals)
	d2x = ForwardDiff.derivative(k -> ForwardDiff.derivative(solve_k, k), 0.5)
	d2x_dense = ForwardDiff.derivative(k -> ForwardDiff.derivative(kk -> to_dense(K(kk)) \ b, k), 0.5)
	@test d2x ≈ d2x_dense atol = 1e-9
	# matrix and right-hand side both depend on the parameter; a dual right-hand side only
	@test ForwardDiff.derivative(k -> spsolve(K(k), k .* B), 0.5) ≈ ForwardDiff.derivative(k -> to_dense(K(k)) \ (k .* B), 0.5)
	@test ForwardDiff.derivative(k -> spsolve(K(0.5), k .* b), 2.0) ≈ to_dense(K(0.5)) \ b
end

@testset "spsolve: lower precisions and integers use the sparse LU" begin
	n = 2000
	A = spdiagm(-1 => fill(-1.0f0, n - 1), 0 => fill(2.0f0, n), 1 => fill(-1.0f0, n - 1))
	b = ones(Float32, n)
	x = spsolve(A, b)
	@test eltype(x) == Float32
	@test x ≈ Float32.(Matrix{Float64}(A) \ Float64.(b))
	@test @allocated(spsolve(A, b)) < n^2  # a dense copy would take 4n² bytes or more
	Ai = sparse([2 -1 0; -1 2 -1; 0 -1 2])
	@test spsolve(Ai, [1, 0, 1]) ≈ [1.0, 1.0, 1.0]
	@test eltype(spsolve(Ai, [1, 0, 1])) == Float64
	xb = spsolve(sparse(BigFloat.(Ai)), BigFloat[1, 0, 1])  # other number types: dense LU
	@test eltype(xb) == BigFloat
	@test xb ≈ [1, 1, 1]
end
