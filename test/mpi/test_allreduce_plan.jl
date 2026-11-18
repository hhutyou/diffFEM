# Port of tatva/tests/test_allreduce_plan.py — run with 2 ranks:
#     mpiexec -n 2 julia --project test/mpi/test_allreduce_plan.jl

using Test, MPI, Tatva, SparseArrays

MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
@assert MPI.Comm_size(comm) == 2 "this test needs exactly 2 MPI ranks"

@testset "allreduce plan allgather (rank $rank)" begin
	plan = AllreducePlan(6; comm = comm)
	allgather = make_allgather(plan)
	x_owned = rank == 0 ? ones(plan.local_size) : 40.0 .* ones(plan.local_size)
	@test allgather(x_owned) == [1.0, 1.0, 1.0, 40.0, 40.0, 40.0]
end

@testset "allreduce plan gradient (rank $rank)" begin
	plan = AllreducePlan(6; comm = comm)
	grad_fn = make_allreduce_owned(plan, identity)
	u_full = rank == 0 ? [0.0, 1.0, 2.0, 3.0, 0.0, 0.0] : [0.0, 0.0, 0.25, 0.25, 4.0, 5.0]
	expected = [0.0, 1.0, 2.25, 3.25, 4.0, 5.0]
	@test grad_fn(u_full) == (rank == 0 ? expected[1:3] : expected[4:6])
end

@testset "allreduce plan Hessian (rank $rank)" begin
	pattern = sparse(1:5, 1:5, fill(true, 5), 5, 5)
	data = rank == 0 ? [1.0, 1.0, 1.0, 1.0, 0.0] : [0.0, 0.0, 0.0, 1.0, 1.0]
	cm = ColoredMatrix(sparse(1:5, 1:5, data, 5, 5))  # explicit zeros are kept
	@test length(cm.data) == 5
	plan = AllreducePlan(5, pattern; comm = comm)
	hessian_fn = make_allreduce_owned(plan, u -> cm; is_hessian = true)
	K = hessian_fn(zeros(6))
	@test K isa ColoredMatrix
	if rank == 0
		@test K.data == [1.0, 1.0, 1.0]
		@test K.indices == [1, 2, 3]
		@test K.indptr == [1, 2, 3, 4]
	else
		@test K.data == [2.0, 1.0]
		@test K.indices == [4, 5]
		@test K.indptr == [1, 2, 3]
	end
end

MPI.Finalize()
