# Port of tatva/tests/test_periodic_mpi.py — run with 2 ranks:
#     mpiexec -n 2 julia --project test/mpi/test_periodic_mpi.jl

using Test, MPI, Tatva, SparseArrays

MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
@assert MPI.Comm_size(comm) == 2 "this test needs exactly 2 MPI ranks"

@testset "PeriodicMPI augments the sparsity when the slave is a ghost (rank $rank)" begin
	# 3 nodes; rank 0 owns node 1 (ghosts 2, 3), rank 1 owns 2 and 3 (ghost 1); node 2 is
	# periodic with master node 3
	l2g, n_owned = rank == 0 ? ([1, 2, 3], 1) : ([2, 3, 1], 2)
	p = PartitionInfo(l2g, n_owned)
	MyState = Compound(:MyState, :u => field((1, length(l2g)); field_type = NODAL);
		mesh = Mesh(zeros(1, length(l2g))), partition_info = p, comm = comm)
	layout = get_layout(MyState)

	cond = PeriodicMPI([2], [3], layout; comm = comm)  # natural global dofs
	lifter = Lifter(MyState.size, cond)
	layout_aug, lifter_aug = adapt_layout(lifter, layout, comm)
	@test lifter_aug.size_reduced == lifter.size - 1

	if rank == 0
		# local coupling between nodes 1 and 2 (node 2 is a ghost slave here)
		sparsity = sparse([1, 1, 2, 2], [1, 2, 1, 2], fill(true, 4), 3, 3)
		aug = augment_sparsity(lifter_aug, sparsity)
		@test aug[1, 3]  # the master (local 3) inherits the coupling with node 1
		@test aug[3, 1]
	end

	# lifting copies the master value into the slave
	u = lift_from_zeros(lifter_aug, collect(1.0:lifter_aug.size_reduced))
	slave = findfirst(==(2), l2g)
	master = findfirst(==(3), l2g)
	@test u[slave] == u[master]
end

@testset "PeriodicMPI with a master missing on one rank only (rank $rank)" begin
	# 4 nodes; rank 0 holds node 1 (owned) and node 2 (ghost), rank 1 owns nodes 2, 3, 4.
	# Node 2 is periodic with master node 4, which rank 0 does not hold: only rank 0 needs an
	# extra ghost, but both ranks must take part in rebuilding the layout (no hang)
	l2g, n_owned = rank == 0 ? ([1, 2], 1) : ([2, 3, 4], 3)
	p = PartitionInfo(l2g, n_owned)
	MyState = Compound(:MyState, :u => field((1, length(l2g)); field_type = NODAL);
		mesh = Mesh(zeros(1, length(l2g))), partition_info = p, comm = comm)
	layout = get_layout(MyState)
	lifter = Lifter(MyState.size, PeriodicMPI([2], [4], layout; comm = comm))
	layout_red, lifter_aug = adapt_layout(lifter, layout, comm)
	@test lifter_aug.size == length(l2g) + (rank == 0 ? 1 : 0)
	@test lifter_aug.size_reduced == lifter_aug.size - 1
	@test layout_red.n_global == 3  # nodes 1, 3 and 4 are free
	# the slave takes the value of node 4's free dof (solver-global index 3) on both ranks
	u = lift_from_zeros(lifter_aug, Float64.(layout_red.local_to_global))
	@test MPI.Allgather(u[findfirst(==(2), l2g)], comm) == [3.0, 3.0]
end

MPI.Finalize()
