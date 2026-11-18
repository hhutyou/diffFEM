# Port of tatva/tests/test_compound_mpi.py — run with 2 ranks:
#     mpiexec -n 2 julia --project test/mpi/test_compound_mpi.jl
# Natural global node and dof indices are 1-based (tatva's + 1).

using Test, MPI, Tatva

MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
@assert MPI.Comm_size(comm) == 2 "this test needs exactly 2 MPI ranks"

@testset "global indices (rank $rank)" begin
	# 4 global nodes; the field `l` lives on the subset of global nodes {2, 4}
	l2g, n_owned, l_nodes = rank == 0 ? ([1, 2, 3], 2, [2]) : ([3, 4, 2], 2, [3, 2])
	p = PartitionInfo(l2g, n_owned)
	MyState = Compound(:MyState,
		:u => field((2, AUTO)),
		:l => field((1, AUTO); field_type = Nodal(node_ids = l_nodes));
		mesh = Mesh(zeros(1, length(l2g))), partition_info = p, comm = comm,
	)
	g = global_indices(MyState)
	# full field u: global shape (2, 4)
	@test g.u[1] == [1, 2]
	@test g.u[2, 4] == [8]
	# subset field l: global shape (1, 2) over global nodes [2, 4], after the 8 dofs of u
	@test g.l[2] == [9]
	@test g.l[4] == [10]
	@test g.l[:] == [9, 10]
	@test g.l[2:2] == [9]   # global node ids 2..2
	@test g.l[3:4] == [10]
	@test g.l[1:2] == [9]
	@test_throws ArgumentError g.l[1:2:4]  # stepped slices of a subset are not supported
	@test_throws BoundsError g.l[1]        # global node 1 is not in the subset
end

@testset "stacked global indices and gathering (rank $rank)" begin
	# 3 global nodes: rank 0 owns 1 and 2, rank 1 owns 3
	l2g, n_owned = rank == 0 ? ([1, 2, 3], 2) : ([3, 1, 2], 1)
	p = PartitionInfo(l2g, n_owned)
	MyState = Compound(:MyState,
		:u => field((1, AUTO); field_type = Nodal()),
		:v => field((2, AUTO); field_type = Nodal());
		mesh = Mesh(zeros(1, length(l2g))), partition_info = p, comm = comm,
	)
	# u and v are stacked node by node: node k has dofs 3(k-1) + [1 | 2, 3]
	g = global_indices(MyState)
	@test g.u[:] == [1, 4, 7]
	@test g.v[:] == [2, 3, 5, 6, 8, 9]
	@test g.u[1, 2] == [4]
	@test g.v[2, 3] == [9]

	state = MyState()
	if rank == 0
		state = set(state, :u, [10.0 20.0 0.0])
		state = set(state, :v, [1.0 3.0 0.0; 2.0 4.0 0.0])
	else
		state = set(state, :u, [30.0 0.0 0.0])  # local node 1 is global node 3
		state = set(state, :v, [5.0 0.0 0.0; 6.0 0.0 0.0])
	end
	gv = global_view(state)
	@test gv.u == [10.0 20.0 30.0]
	@test gv.v == [1.0 3.0 5.0; 2.0 4.0 6.0]
end

MPI.Finalize()
