# Port of tatva/tests/test_linalg.py.
#
# The numerical contract of `Tatva.Linalg` is ported one to one. tatva's HLO tests (counting
# `dot`/cuBLAS/LU ops in the compiled XLA program, platform-dependent lowering) have no
# meaning without XLA; their Julia counterpart — element kernels must be allocation free and
# type stable — is in test_element.jl / test_operator.jl.

using Test, Tatva, LinearAlgebra, Random, ForwardDiff, StaticArrays
using Tatva.Linalg: contract, einsum
const TL = Tatva.Linalg

well_conditioned(d, seed = 0) = Matrix{Float64}(I, d, d) + 0.1 * randn(MersenneTwister(seed), d, d)

# reference numpy.tensordot(A, B, axes=1): contract A's last axis with B's first
function tensordot_ref(A, B)
	sa, sb = size(A), size(B)
	Am = reshape(A, :, sa[end])
	Bm = reshape(B, sb[1], :)
	R = Am * Bm
	shape = (sa[1:(end-1)]..., sb[2:end]...)
	return isempty(shape) ? R[1] : reshape(R, shape)
end

#--------------------------------------------------------------------------------
# inv
#--------------------------------------------------------------------------------

@testset "inv matches LinearAlgebra.inv (d=$d)" for d in 1:3
	J = well_conditioned(d)
	@test TL.inv(J) ≈ inv(J)
	@test TL.inv(J) * J ≈ I(d) atol = 1e-12
end

@testset "inv maps over a batch (d=$d)" for d in 2:3
	rng = MersenneTwister(1)
	stack = [Matrix{Float64}(I, d, d) + 0.1 * randn(rng, d, d) for _ in 1:16]
	@test all(map(TL.inv, stack) .≈ map(inv, stack))
end

@testset "inv rejects anything that is not a single matrix: $shape" for shape in ((4, 2, 2), (3, 3, 3), (2,))
	@test_throws "single matrix" TL.inv(zeros(shape))
end

@testset "inv keeps static matrices static" begin
	J = SMatrix{2, 2}(well_conditioned(2))
	@test TL.inv(J) isa SMatrix{2, 2}
	@test TL.inv(J) ≈ inv(J)
	@test TL.inv(well_conditioned(4)) ≈ inv(well_conditioned(4))  # fallback beyond 3×3
end

#--------------------------------------------------------------------------------
# det
#--------------------------------------------------------------------------------

@testset "det matches LinearAlgebra.det (d=$d)" for d in 1:3
	J = well_conditioned(d)
	@test TL.det(J) ≈ det(J)
	@test TL.det(J) isa Float64  # a scalar, not a 1×1 array
	@test TL.det(J) * TL.det(TL.inv(J)) ≈ 1.0
end

@testset "det maps over a batch (d=$d)" for d in 2:3
	rng = MersenneTwister(1)
	stack = [Matrix{Float64}(I, d, d) + 0.1 * randn(rng, d, d) for _ in 1:16]
	@test map(TL.det, stack) ≈ map(det, stack)
end

@testset "det rejects anything that is not a single matrix: $shape" for shape in ((4, 2, 2), (3, 3, 3), (2,))
	@test_throws "single matrix" TL.det(zeros(shape))
end

#--------------------------------------------------------------------------------
# tensordot
#--------------------------------------------------------------------------------

const ACCEPTED_SHAPES = [
	((3,), (3,)),
	((3,), (3, 2)),
	((3,), (3, 2, 2)),
	((2, 3), (3,)),
	((2, 3), (3, 2)),
	((2, 3), (3, 2, 2)),
	((2, 3), (3, 4, 2, 2)),
]

@testset "_dot_broadcast is tensordot(A, B, axes=1): $sa × $sb" for (sa, sb) in ACCEPTED_SHAPES
	rng = MersenneTwister(2)
	A, B = rand(rng, sa...), rand(rng, sb...)
	@test TL._dot_broadcast(A, B) ≈ tensordot_ref(A, B)
end

@testset "_dot_broadcast is not a batched matmul for rank-3 operands" begin
	rng = MersenneTwister(6)
	A, B = rand(rng, 2, 3), rand(rng, 3, 3, 4)
	@test size(TL._dot_broadcast(A, B)) == (2, 3, 4)
	@test TL._dot_broadcast(A, B) ≈ tensordot_ref(A, B)
end

@testset "_dot_broadcast maps over a batch" begin
	rng = MersenneTwister(3)
	A, B = rand(rng, 16, 2, 3), rand(rng, 16, 3, 4)
	batched = stack([TL._dot_broadcast(A[e, :, :], B[e, :, :]) for e in 1:16]; dims = 1)
	@test batched ≈ einsum("epq,eqr->epr", A, B)
end

@testset "_dot_broadcast rejects a stack of matrices" begin
	@test_throws "single vector or matrix" TL._dot_broadcast(zeros(5, 2, 3), zeros(3, 4))
end

#--------------------------------------------------------------------------------
# contract
#--------------------------------------------------------------------------------

const CONTRACT_CASES = [
	("in,nj->ij", (2, 4), (4, 2), "in,nj->ij"),
	("ij,jn->in", (3, 3), (3, 8), "ij,jn->in"),
	("in,n...->i...", (2, 4), (4,), "in,n->i"),
	("in,n...->i...", (2, 4), (4, 3), "in,nc->ic"),
	("in,n...->i...", (2, 4), (4, 3, 3), "in,ncd->icd"),
	("n,n...->...", (4,), (4,), "n,n->"),
	("n,n...->...", (4,), (4, 3), "n,nc->c"),
	("n,nj->j", (4,), (4, 2), "n,nj->j"),
	("...ir,jr->...ij", (3, 2, 2), (2, 2), "cir,jr->cij"),
]

@testset "contract computes what the spec says: $spec $sa $sb" for (spec, sa, sb, ref) in CONTRACT_CASES
	rng = MersenneTwister(9)
	A, B = rand(rng, sa...), rand(rng, sb...)
	@test contract(spec, A, B) ≈ einsum(ref, A, B)
end

@testset "contract is indifferent to the index letters" begin
	rng = MersenneTwister(10)
	A, B = rand(rng, 2, 4), rand(rng, 4, 3)
	@test contract("in,nj->ij", A, B) ≈ contract("pq,qr->pr", A, B)
	@test contract("ab,bc->ac", A, B) ≈ A * B
	@test TL._canonicalise("in,nj->ij") == "ab,bc->ac"
	@test TL._canonicalise("in,n...->i...") == "ab,b...->a..."
end

@testset "contract refuses an unmeasured contraction" begin
	A, B = zeros(2, 3), zeros(3, 4)
	@test_throws "no implementation" contract("ij,jk->ijk", A, B)  # no contraction at all
	@test_throws "no implementation" contract("ij,jk->ki", A, B)   # transposed output
end

@testset "contract rejects a batch axis: $spec" for spec in ("in,nj->ij", "n,n...->...", "in,n...->i...")
	@test_throws "ranks" contract(spec, zeros(5, 2, 3), zeros(3, 4))
end

@testset "contract rejects a mismatched contraction axis" begin
	@test_throws DimensionMismatch contract("in,nj->ij", zeros(2, 3), zeros(4, 2))
	@test_throws "contracts A's last axis" contract("in,nj->ij", zeros(2, 3), zeros(4, 2))
end

@testset "hessian transform rejects invalid operand ranks" begin
	@test_throws "ranks" contract("...ir,jr->...ij", zeros(2), zeros(2, 2))
	@test_throws "ranks" contract("...ir,jr->...ij", zeros(2, 2), zeros(3, 2, 2))
end

@testset "contract rejects an unparseable spec: $spec" for spec in ("ij->i", "ij,jk", "ij,jk->ik->ik", "i1,1k->ik")
	@test_throws ArgumentError contract(spec, zeros(2, 3), zeros(3, 4))
end

@testset "every contract spec used in the package is in the table" begin
	src = joinpath(pkgdir(Tatva), "src")
	literals = Set{String}()
	for (root, _, files) ∈ walkdir(src), f ∈ files
		endswith(f, ".jl") || continue
		for m ∈ eachmatch(r"contract\(\s*\"([^\"]+)\"", read(joinpath(root, f), String))
			push!(literals, m.captures[1])
		end
	end
	for spec ∈ literals
		@test haskey(TL._known_implementations, TL._canonicalise(spec))
	end
end

const DIFF_CASES = [
	("in,nj->ij", (2, 4), (4, 2), "in,nj->ij"),
	("in,n...->i...", (2, 4), (4, 3), "in,nc->ic"),
	("n,n...->...", (4,), (4, 3), "n,nc->c"),
	("n,nj->j", (4,), (4, 2), "n,nj->j"),
	("...ir,jr->...ij", (3, 2, 2), (2, 2), "cir,jr->cij"),
]

@testset "contract differentiates like the einsum it names: $spec" for (spec, sa, sb, ref) in DIFF_CASES
	rng = MersenneTwister(12)
	A, B = rand(rng, sa...), rand(rng, sb...)
	mine_A = ForwardDiff.gradient(a -> sum(contract(spec, a, B)), A)
	mine_B = ForwardDiff.gradient(b -> sum(contract(spec, A, b)), B)
	@test mine_A ≈ ForwardDiff.gradient(a -> sum(einsum(ref, a, B)), A)
	@test mine_B ≈ ForwardDiff.gradient(b -> sum(einsum(ref, A, b)), B)
end

const BATCH_CASES = [
	("in,nj->ij", (2, 4), (4, 2), "ein,enj->eij"),
	("in,n...->i...", (2, 4), (4, 3), "ein,enc->eic"),
	("n,n...->...", (4,), (4, 3), "en,enc->ec"),
	("n,nj->j", (4,), (4, 2), "en,enj->ej"),
	("...ir,jr->...ij", (3, 2, 2), (2, 2), "ecir,ejr->ecij"),
]

@testset "contract maps over elements: $spec" for (spec, sa, sb, ref) in BATCH_CASES
	rng = MersenneTwister(13)
	A, B = rand(rng, 16, sa...), rand(rng, 16, sb...)
	colons(x) = ntuple(_ -> Colon(), ndims(x) - 1)
	batched = stack([contract(spec, A[e, colons(A)...], B[e, colons(B)...]) for e in 1:16]; dims = 1)
	@test batched ≈ einsum(ref, A, B)
end

#--------------------------------------------------------------------------------
# einsum: the drop-in wrapper
#--------------------------------------------------------------------------------

@testset "einsum: contract path for a measured spec, general path otherwise: $spec" for (spec, shapes, path) in (
	("in,nj->ij", [(2, 4), (4, 2)], :contract),
	("pq,qr->pr", [(2, 4), (4, 2)], :contract),
	("in,n...->i...", [(2, 4), (4, 3)], :contract),
	("n,n...->...", [(4,), (4, 3)], :contract),
	("...ir,jr->...ij", [(3, 2, 2), (2, 2)], :contract),
	("eq,q->eq", [(64, 4), (4,)], :general),
	("ij,jk->ki", [(2, 3), (3, 4)], :general),
	("ij->ji", [(2, 3)], :general),
	("ij,jk", [(2, 3), (3, 4)], :general),
	("ij,jk,kl->il", [(2, 3), (3, 4), (4, 2)], :general),
)
	rng = MersenneTwister(14)
	arrays = [rand(rng, s...) for s in shapes]
	canonical = try
		TL._canonicalise(spec)
	catch
		nothing
	end
	took_contract = canonical !== nothing && haskey(TL._known_implementations, canonical) && length(arrays) == 2
	@test took_contract == (path === :contract)
	result = einsum(spec, arrays...)
	if spec == "eq,q->eq"
		@test result ≈ arrays[1] .* arrays[2]'
	elseif spec == "ij,jk->ki"
		@test result ≈ (arrays[1] * arrays[2])'
	elseif spec == "ij->ji"
		@test result ≈ permutedims(arrays[1])
	elseif spec == "ij,jk"  # implicit output: indices that appear once, sorted
		@test result ≈ arrays[1] * arrays[2]
	elseif spec == "ij,jk,kl->il"
		@test result ≈ arrays[1] * arrays[2] * arrays[3]
	elseif spec in ("in,nj->ij", "pq,qr->pr")
		@test result ≈ arrays[1] * arrays[2]
	end
end

@testset "einsum interleaved calling convention" begin
	rng = MersenneTwister(15)
	A, B = rand(rng, 2, 3), rand(rng, 3, 4)
	@test einsum(A, [0, 1], B, [1, 2], [0, 2]) ≈ A * B
end

#--------------------------------------------------------------------------------
# line elements measure arc length in 2D and 3D (a helix)
#--------------------------------------------------------------------------------

@testset "line elements measure arc length: $E in $(dim)D" for (E, npe) in ((Line2, 2), (Line3, 3)), dim in (2, 3)
	n_elem, turns, radius, pitch = 400, 1.0, 2.0, 3.0
	n_nodes = n_elem * (npe - 1) + 1
	s = range(0.0, 2π * turns; length = n_nodes)
	rows = [radius .* cos.(s), radius .* sin.(s)]
	dim == 3 && push!(rows, pitch .* s ./ (2π * turns))
	coords = permutedims(reduce(hcat, rows))  # (dim, n_nodes)
	elements = if npe == 2
		permutedims(hcat(1:n_elem, 2:(n_elem+1)))
	else  # Line3 stores (end, end, mid)
		first = 1 .+ 2 .* (0:(n_elem-1))
		permutedims(hcat(first, first .+ 2, first .+ 1))
	end
	op = Operator(Mesh(coords, elements), E())
	len = sum(get_integration_weights(op))
	exact = turns * sqrt((2π * radius)^2 + (dim == 3 ? pitch : 0.0)^2)
	@test abs(len - exact) / exact < 1e-4
end
