# Port of tatva/tests/test_sparse_tracer.py: sparsity tracing end to end through the FEM
# stack (Operator, Compound, Lifter).

using Test, Tatva, SparseArrays, LinearAlgebra, StaticArrays, ForwardDiff, Random

function nz_set(m)
	rows, cols, vals = findnz(sparse(m))
	keep = vals .!= 0
	return Set{Tuple{Int, Int}}(zip(rows[keep], cols[keep]))
end

"""Structured tetrahedral mesh of a box (6 tets per hexahedral cell), as in tatva's test."""
function box_tet_mesh(lengths, nb_elems)
	Lx, Ly, Lz = lengths
	nx, ny, nz = nb_elems
	xs = range(-Lx / 2, Lx / 2; length = nx + 1)
	ys = range(-Ly / 2, Ly / 2; length = ny + 1)
	zs = range(0.0, Lz; length = nz + 1)
	sy, sz = nx + 1, (nx + 1) * (ny + 1)
	coords = zeros(3, sz * (nz + 1))
	for k ∈ 0:nz, j ∈ 0:ny, i ∈ 0:nx
		coords[:, i+j*sy+k*sz+1] .= (xs[i+1], ys[j+1], zs[k+1])
	end
	tets = NTuple{4, Int}[]
	for k ∈ 0:(nz-1), j ∈ 0:(ny-1), i ∈ 0:(nx-1)
		n0 = i + j * sy + k * sz
		n1, n2 = n0 + 1, n0 + sy
		n3, n4 = n2 + 1, n0 + sz
		n5, n6 = n4 + 1, n4 + sy
		n7 = n6 + 1
		push!(tets, (n0, n1, n3, n7), (n0, n1, n7, n5), (n0, n5, n7, n4), (n0, n3, n2, n7), (n0, n2, n6, n7), (n0, n6, n4, n7))
	end
	return Mesh(coords, [t[a] + 1 for a in 1:4, t in tets])
end

"""Line2 mesh of `n_nodes` nodes along the x-axis of 2D space."""
function line_operator(n_nodes; length = 1.0)
	x = range(0.0, length; length = n_nodes)
	coords = permutedims(hcat(x, zeros(n_nodes)))
	elements = permutedims(hcat(1:(n_nodes-1), 2:n_nodes))
	return Operator(Mesh(coords, elements), Line2())
end

#--------------------------------------------------------------------------------
# energy functionals (∂²E/∂u²)
#--------------------------------------------------------------------------------

@testset "3D vector field: neo-Hookean energy with a Dirichlet lifter" begin
	mesh = box_tet_mesh((10.0, 1.0, 1.0), (10, 2, 2))
	Solution = Compound(:Solution, :u => field((3, AUTO)); mesh = mesh)
	op = Operator(mesh, Tetrahedron4())
	μ, λ = 500.0, 1000.0
	function ψ(∇u)
		F = I + ∇u
		J = det(F)
		C = F' * F
		return (μ / 2) * (tr(C) - 3 - 2 * log(J)) + (λ / 2) * log(J)^2
	end
	x = mesh.coords[1, :]
	fixed_nodes = findall(isapprox.(x, minimum(x)))
	load_nodes = findall(isapprox.(x, maximum(x)))
	lifter = Lifter(Solution.size, Fixed(Solution.u[:, fixed_nodes]), Fixed(Solution.u[3, load_nodes], 1.0))

	total_energy(sol) = integrate(op, ψ.(grad(op, sol.u)))
	total_energy_free(u_free, lf) = total_energy(Solution(lift_from_zeros(lf, u_free)))

	reduced = adapt_sparsity(lifter, pattern_from_mesh(mesh, 3))
	traced = pattern_from_energy(total_energy_free, lifter.size_reduced, lifter)
	# the tracer must not miss any coupling of the element pattern
	@test nz_set(reduced) ⊆ nz_set(traced)
	@test nz_set(traced) == nz_set(reduced)  # and for this energy it is exactly that pattern
end

@testset "periodic + Dirichlet constraints: traced ⊆ tatva's augment/reduce pipeline" begin
	mesh = unit_square(4, 4)
	op = Operator(mesh, Tri3())
	x, y = mesh.coords[1, :], mesh.coords[2, :]
	bottom = findall(y .== minimum(y))
	left = findall(x .== 0.0)
	left = left[sortperm(y[left])]
	right = findall(x .== 1.0)
	right = right[sortperm(y[right])]

	Solution = Compound(:Solution, :u => field((2, AUTO)); mesh = mesh)
	lifter = Lifter(Solution.size, Fixed(Solution.u[:, bottom], 0.0), Periodic(Solution.u[:, left], Solution.u[:, right]))

	full = pattern_from_compound(Solution)
	@test full == pattern_from_mesh(mesh, 2)
	expected = reduce_sparsity(lifter, augment_sparsity(lifter, full))

	μ, λ = 1.0, 1.0
	ψ(∇u) = (F = I + ∇u; J = det(F); (μ / 2) * (tr(F' * F) - 2 - 2 * log(J)) + (λ / 2) * log(J)^2)
	energy_free(u_free, lf) = integrate(op, ψ.(grad(op, Solution(lift_from_zeros(lf, u_free)).u)))
	actual = pattern_from_energy(energy_free, lifter.size_reduced, lifter)
	@test nz_set(actual) ⊆ nz_set(expected)
	# and no false negatives against the dense reduced Hessian
	ur = 0.01 .* randn(MersenneTwister(2), lifter.size_reduced)
	H = ForwardDiff.hessian(u -> energy_free(u, lifter), ur)
	@test Set(Tuple.(findall(abs.(H) .> 1e-12))) ⊆ nz_set(actual)
end

@testset "Lagrange multipliers (adhesion): no false negatives" begin
	Lx, Ly, Nx, Ny = 6.0, 2.0, 4, 3
	xs = range(0.0, Lx; length = Nx)
	ys = range(0.0, Ly; length = Ny)
	coords = zeros(2, Nx * Ny)
	for i ∈ 0:(Nx-1), j ∈ 0:(Ny-1)
		coords[:, i*Ny+j+1] .= (xs[i+1], ys[j+1])
	end
	tris = NTuple{3, Int}[]
	for i ∈ 0:(Nx-2), j ∈ 0:(Ny-2)
		n0, n1 = i * Ny + j, (i + 1) * Ny + j
		n2, n3 = i * Ny + j + 1, (i + 1) * Ny + j + 1
		push!(tris, (n0, n1, n2), (n1, n3, n2))
	end
	elements = [t[a] + 1 for a in 1:3, t in tris]
	line_elements = [i * Ny + 1 + a * Ny for a in 0:1, i in 0:(Nx-2)]  # bottom edge (y = 0)
	mesh = Mesh(coords, elements)
	op = Operator(mesh, Tri3())
	op_line = Operator(Mesh(coords, line_elements), Line2())
	n_line = size(line_elements, 2)

	μ, λ = 0.5, 1.0
	ψ(∇u) = (F = I + ∇u; J = det(F); μ / 2 * (tr(F' * F) - 2) - μ * log(J) + (λ / 2) * log(J)^2)
	Solution = Compound(:Solution, :u_rigid => field((3,)), :u => field((2, Nx * Ny)), :lm => field((n_line,)))

	function constraint_potential(lm, u, active)
		uq = evaluate(op_line, u)
		density = [active[e] ? lm[e] * uq[q, e][2] : 0.5 * lm[e]^2 for q in axes(uq, 1), e in axes(uq, 2)]
		return integrate(op_line, density)
	end
	function total_lagrangian(z, active)
		s = Solution(z)
		return integrate(op, ψ.(grad(op, s.u))) + constraint_potential(s.lm, s.u, active)
	end
	active = trues(n_line)
	energy_fn = z -> total_lagrangian(z, active)

	z = 0.04 .* rand(MersenneTwister(3), Solution.size) .- 0.02
	reference = Set(Tuple.(findall(abs.(ForwardDiff.hessian(energy_fn, z)) .> 1e-12)))
	traced = pattern_from_energy(energy_fn, Solution.size)
	cm = ColoredMatrix(traced)  # building a colored matrix from a traced pattern works
	@test maximum(cm.colors) >= 1
	@test reference ⊆ nz_set(traced)
end

@testset "adjacent nonlinear blocks (tatva: jax.checkpoint / remat2)" begin
	M = 8
	coupled_block(ε) = 0.5 * ε^2 + 0.25 * ε^4
	energy(U) = sum(coupled_block(U[i] - U[i+1]) for i in 1:(M-1))
	H = ForwardDiff.hessian(energy, randn(MersenneTwister(4), M))
	reference = abs.(H) .> 1e-12
	pat = pattern_from_energy(energy, M)
	@test Set(Tuple.(findall(reference))) ⊆ nz_set(pat)
	@test nnz(pat) == count(reference) == 3M - 2  # exactly tridiagonal
end

#--------------------------------------------------------------------------------
# virtual work (K = ∂R/∂u = ∂²G/∂v∂u)
#--------------------------------------------------------------------------------

function vw_reference(vw, n, u_dummy)
	R(u) = ForwardDiff.gradient(w -> vw(w, u), zeros(eltype(u), n))
	K = ForwardDiff.jacobian(R, u_dummy)
	return sparse(abs.(K) .> 1e-12)
end

function assert_traced_matches(pat, K_pat, lifter; exact = true)
	free = lifter.free_dofs
	K_reduced = K_pat[free, free]
	if exact
		@test nz_set(pat) == nz_set(K_pat)
		@test nz_set(reduce_sparsity(lifter, pat)) == nz_set(K_reduced)
	else
		@test nz_set(K_pat) ⊆ nz_set(pat)
		@test nz_set(K_reduced) ⊆ nz_set(reduce_sparsity(lifter, pat))
	end
end

pointwise(f, fields...) = map(f, fields...)

@testset "advection–diffusion (non-symmetric) on a Line2 mesh" begin
	op = line_operator(8)
	D, V = 1.0, 1.5
	Field = Compound(:Field, :u => field((1, AUTO)); mesh = op.mesh)
	function g_fn(w, u)
		u_n, w_n = Field(u).u, Field(w).u
		du, dw, w_q = grad(op, u_n), grad(op, w_n), evaluate(op, w_n)
		return sum(integrate(op, pointwise((a, b, c) -> D .* a .* b .+ V .* a .* c, du, dw, w_q)))
	end
	lifter = Lifter(Field.size, Fixed(Field.u[:, [1]]))
	pat = pattern_from_virtual_work(g_fn, Field.size, :u, :w)
	K_pat = vw_reference(g_fn, Field.size, collect(0.0:(Field.size-1)))
	assert_traced_matches(pat, K_pat, lifter)
end

@testset "nonlinear reaction–diffusion (w u³)" begin
	N = 6
	op = line_operator(N)
	Field = Compound(:Field, :u => field((1, AUTO)); mesh = op.mesh)
	function g_fn(w, u)
		u_n, w_n = Field(u).u, Field(w).u
		integrand = pointwise((a, b, wq, uq) -> a .* b .+ wq .* uq .^ 3, grad(op, u_n), grad(op, w_n), evaluate(op, w_n), evaluate(op, u_n))
		return sum(integrate(op, integrand))
	end
	lifter = Lifter(Field.size, Fixed(Field.u[:, [1]]))
	pat = pattern_from_virtual_work(g_fn, Field.size, :u, :w)
	K_pat = vw_reference(g_fn, Field.size, collect(1.0:N))  # away from the trivial u = 0
	assert_traced_matches(pat, K_pat, lifter)
end

@testset "mixed velocity–pressure saddle point (stacked compound)" begin
	N = 5
	op = line_operator(N)
	Mixed = Compound(:Mixed, :v => field((1, AUTO)), :p => field((1, AUTO)); mesh = op.mesh)
	function g_fn(test, trial)
		v, p = Mixed(trial)
		w, q = Mixed(test)
		dv, dw = grad(op, v), grad(op, w)
		p_q, q_q = evaluate(op, p), evaluate(op, q)
		elastic = sum(integrate(op, pointwise((a, b) -> a .* b, dw, dv)))
		coupling1 = sum(integrate(op, pointwise((a, b) -> a .* b, dw, p_q)))
		coupling2 = sum(integrate(op, pointwise((a, b) -> a .* b, q_q, dv)))
		return elastic + coupling1 + coupling2
	end
	lifter = Lifter(Mixed.size, Fixed(Mixed.v[:, [1]]))
	pat = pattern_from_virtual_work(g_fn, Mixed.size, :trial, :test)
	K_pat = vw_reference(g_fn, Mixed.size, collect(0.0:(Mixed.size-1)))
	assert_traced_matches(pat, K_pat, lifter; exact = false)
end

@testset "unsymmetric two-field coupling" begin
	N = 6
	op = line_operator(N)
	TwoField = Compound(:TwoField, :a => field((1, AUTO)), :b => field((1, AUTO)); mesh = op.mesh)
	function g_fn(test, trial)
		a, b = TwoField(trial)
		wa, wb = TwoField(test)
		prod_(x, y) = sum(integrate(op, pointwise((s, t) -> s .* t, x, y)))
		diffusion = prod_(grad(op, wa), grad(op, a)) + prod_(grad(op, wb), grad(op, b))
		one_way = prod_(evaluate(op, wa), grad(op, b))  # the a-residual depends on b, not vice versa
		return diffusion + one_way
	end
	lifter = Lifter(TwoField.size, Fixed(TwoField.a[:, [1]]))
	pat = pattern_from_virtual_work(g_fn, TwoField.size, :trial, :test)
	K_pat = vw_reference(g_fn, TwoField.size, collect(0.0:(TwoField.size-1)))
	@test nz_set(K_pat) != nz_set(sparse(transpose(K_pat)))
	@test nz_set(pat) != nz_set(sparse(transpose(pat)))
	assert_traced_matches(pat, K_pat, lifter; exact = false)
end

#--------------------------------------------------------------------------------
# external constitutive leaves
#--------------------------------------------------------------------------------

@testset "a constitutive leaf mapped per quadrature point keeps the elastic pattern" begin
	N = 6
	op = line_operator(N)
	Field = Compound(:Field, :u => field((1, AUTO)); mesh = op.mesh)
	stress_qp(strain) = 2.0 .* strain
	g_leaf(w, u) = sum(integrate(op, pointwise(dot, map(stress_qp, grad(op, Field(u).u)), grad(op, Field(w).u))))
	g_elastic(w, u) = sum(integrate(op, pointwise((s, t) -> dot(2.0 .* s, t), grad(op, Field(u).u), grad(op, Field(w).u))))
	ref = pattern_from_virtual_work(g_elastic, Field.size, :u, :w)
	@test nz_set(pattern_from_virtual_work(g_leaf, Field.size, :u, :w)) == nz_set(ref)
end

@testset "an opaque leaf: dense when batched, sparse per quadrature point (tatva: ffi_call)" begin
	N = 6
	op = line_operator(N)
	Field = Compound(:Field, :u => field((1, AUTO)); mesh = op.mesh)
	rve(s) = 2.0 .* s  # stands for an external solver
	g_batched(w, u) = sum(integrate(op, pointwise(dot, opaque(S -> map(rve, S), grad(op, Field(u).u)), grad(op, Field(w).u))))
	g_pointwise(w, u) = sum(integrate(op, pointwise((s, t) -> dot(opaque(rve, s), t), grad(op, Field(u).u), grad(op, Field(w).u))))
	g_elastic(w, u) = sum(integrate(op, pointwise(dot, grad(op, Field(u).u), grad(op, Field(w).u))))
	ref = pattern_from_virtual_work(g_elastic, Field.size, :u, :w)
	dense = pattern_from_virtual_work(g_batched, Field.size, :u, :w)
	@test nz_set(ref) ⊆ nz_set(dense)
	@test nnz(dense) > nnz(ref)
	@test nz_set(pattern_from_virtual_work(g_pointwise, Field.size, :u, :w)) == nz_set(ref)
end
