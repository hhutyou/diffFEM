# Run the MPI tests (ports of tatva's test_*_mpi.py / *_plan.py) on 2 ranks, each file in
# its own `mpiexec` job. Skipped when MPI.jl is not available.

using Test

const MPI_TEST_FILES = [
	"test_exchange_plan.jl",
	"test_allreduce_plan.jl",
	"test_compound_mpi.jl",
	"test_periodic_mpi.jl",
	"test_parallel_assembly.jl",
]

mpi_available = try
	@eval import MPI
	true
catch
	false
end

# a mismatch of collective calls between ranks hangs: kill the job after `timeout` seconds
function run_with_timeout(cmd, timeout)
	proc = run(pipeline(cmd; stdout = stdout, stderr = stderr); wait = false)
	timer = Timer(_ -> process_running(proc) && kill(proc), timeout)
	wait(proc)
	close(timer)
	return success(proc)
end

if mpi_available
	@testset "MPI (2 ranks)" begin
		project = dirname(Base.active_project())
		for file ∈ MPI_TEST_FILES
			path = joinpath(@__DIR__, file)
			cmd = `$(MPI.mpiexec()) -n 2 $(Base.julia_cmd()) --startup-file=no --project=$project $path`
			@test run_with_timeout(cmd, 1200)
		end
	end
else
	@info "MPI.jl not available: skipping the MPI tests"
end
