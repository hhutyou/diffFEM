# Head-to-head with tatva (see tatva_bench.py in this folder): the same 2D linear-elastic
# energy on unit_square(n, n) (Tri3), the same displacement state, median of 5 runs after the
# first call (which includes compilation).
#
#     julia -t 8 --project=<env with Tatva and Enzyme> julia_bench.jl
#
# Routes: global API with ForwardDiff (jacrev / sparse_hessian; O(n_dofs²), small meshes only),
# global gradient with Enzyme (reverse mode), and the element-wise ElementAssembler on one
# thread and on all threads ("all" = energy, gradient and Hessian from one pass). The check
# values must match tatva_bench.py.

using Tatva, LinearAlgebra, SparseArrays, Printf, Statistics
const HAS_ENZYME = Base.find_package("Enzyme") !== nothing
HAS_ENZYME && @eval using Enzyme

const μ, λ = 1.0, 1.0
ψ(∇u) = (ε = (∇u + ∇u') / 2; μ * dot(ε, ε) + λ / 2 * tr(ε)^2)

# (first call, median of the later calls)
function timed(f; repeats = 5)
	first = @elapsed f()
	return first, median([@elapsed(f()) for _ in 1:repeats])
end
ms(t) = t === nothing ? "        -" : @sprintf("%9.2f", 1e3 * t)
run_ms(t) = t === nothing ? ms(nothing) : ms(t[2])

println("Julia ", VERSION, ", ", Threads.nthreads(), " threads")
first_calls = Dict{String, Float64}()
for n ∈ (20, 50, 100, 200, 280)
	mesh = unit_square(n, n)
	op = Operator(mesh, Tri3())
	x, y = mesh.coords[1, :], mesh.coords[2, :]
	U = 0.01 .* permutedims(hcat(sin.(3x) .* y, x .* y .^ 2))  # u = 0.01 (sin(3x) y, x y²)
	u = vec(U)
	E(v) = integrate(op, ψ.(grad(op, reshape(v, 2, :))))
	density(ξ, Ue, Xe) = ψ(grad(op.element, ξ, Ue, Xe))
	asm = ElementAssembler(op, 2)
	asm_t = ElementAssembler(op, 2; threaded = true)
	cm = ColoredMatrix(pattern_from_mesh(mesh, 2))

	t = Dict{String, Any}()
	t["E global"] = timed(() -> E(u))
	t["E element"] = timed(() -> assemble_energy(asm, density, U, mesh.coords))
	t["g ForwardDiff"] = n <= 50 ? timed(() -> jacrev(E)(u)) : nothing
	t["g Enzyme"] = HAS_ENZYME ? timed(() -> jacrev(E; backend = AutoEnzyme())(u)) : nothing
	t["g element"] = timed(() -> assemble_gradient(asm, density, U, mesh.coords))
	t["g element, threads"] = timed(() -> assemble_gradient(asm_t, density, U, mesh.coords))
	t["H ForwardDiff"] = n <= 20 ? timed(() -> sparse_hessian(E, cm)(u)) : nothing
	t["all element"] = timed(() -> assemble(asm, density, U, mesh.coords))
	t["all element, threads"] = timed(() -> assemble(asm_t, density, U, mesh.coords))
	g, K = zeros(length(u)), similar(asm.pattern)
	t["all! element, threads"] = timed(() -> assemble!(g, K, asm_t, density, U, mesh.coords))
	for (k, v) ∈ t
		v === nothing || haskey(first_calls, k) || (first_calls[k] = v[1])
	end

	Ev, gv, Kv = assemble(asm, density, U, mesh.coords)
	n == 20 && @printf("%9s %7s | %18s | %38s | %38s\n", "mesh", "n_dofs", "energy: global / elem",
		"gradient: FD / Enzyme / elem / elem-thr", "Hessian: FD / all elem / thr / thr in place")
	@printf("%4d×%-4d %7d | %s %s | %s %s %s %s | %s %s %s %s   check: E=%.12e |g|=%.12e |K|=%.12e\n",
		n, n, length(u), run_ms(t["E global"]), run_ms(t["E element"]),
		run_ms(t["g ForwardDiff"]), run_ms(t["g Enzyme"]), run_ms(t["g element"]), run_ms(t["g element, threads"]),
		run_ms(t["H ForwardDiff"]), run_ms(t["all element"]), run_ms(t["all element, threads"]), run_ms(t["all! element, threads"]),
		Ev, norm(gv), norm(nonzeros(Kv)))
end
println("\nfirst calls (compilation included; later mesh sizes need no new compilation):")
for (k, v) ∈ sort!(collect(first_calls))
	@printf("  %-22s %9.1f ms\n", k, 1e3 * v)
end
