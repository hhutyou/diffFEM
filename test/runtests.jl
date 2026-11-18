using Test

# Each file is a port of the tatva test module of the same name (see the header of every
# file for what is ported and what has no Julia counterpart). MPI tests run in separate
# processes under `mpiexec` (test/mpi/runtests_mpi.jl) when MPI.jl is available.

const TEST_FILES = [
	"test_element.jl",
	"test_mesh.jl",
	"test_operator.jl",
	"test_linalg.jl",
	"test_sparse.jl",
	"test_operator_projection.jl",
	"test_lifter.jl",
	"test_compound.jl",
	"test_sparse_tracer_core.jl",
	"test_sparse_tracer.jl",
	"test_examples.jl",
	"test_misc.jl",
	"test_assembly.jl",
	"test_solvers.jl",
	"test_extensions.jl",  # loads HyperHessians (switches element Hessians to hyper-duals): last
]

@testset "Tatva.jl" begin
	for file ∈ TEST_FILES
		@testset "$file" begin
			# every file in its own module, so that test-local definitions do not clash
			mod = Module(Symbol(splitext(file)[1]))
			Base.include(mod, joinpath(@__DIR__, file))
		end
	end
	include(joinpath(@__DIR__, "run_enzyme_tests.jl"))
	include(joinpath(@__DIR__, "mpi", "runtests_mpi.jl"))
end
