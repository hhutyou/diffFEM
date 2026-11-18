# Port of tatva/tests/test_exchange_plan.py — run with 2 ranks:
#     mpiexec -n 2 julia --project test/mpi/test_exchange_plan.jl
# Global/natural indices are 1-based (tatva's + 1); ranks stay 0-based.

using Test, MPI, Tatva, SparseArrays

MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
@assert MPI.Comm_size(comm) == 2 "this test needs exactly 2 MPI ranks"

# two nodes: each rank owns one and has the other as a ghost
two_node_partition() = rank == 0 ? PartitionInfo([1, 2], 1) : PartitionInfo([2, 1], 1)
mock_mesh(p) = Mesh(zeros(1, length(p.nodes_local_to_global)))

@testset "exchange plan layout (rank $rank)" begin
	p = two_node_partition()
	MyState = Compound(:MyState,
		:u => field((2, 2); field_type = NODAL),   # 2 dofs per node, 2 nodes
		:s => field((1,); field_type = SHARED),
		:v => field((1,); field_type = LOCAL);
		mesh = mock_mesh(p), partition_info = p, comm = comm,
	)
	# owned: rank 0 → 2 (u) + 1 (s) + 1 (v) = 4, rank 1 → 2 (u) + 1 (v) = 3; global 7
	layout_reduced, _ = adapt_layout(Lifter(MyState.size), get_layout(MyState), comm)
	plan = ExchangePlan(layout_reduced; comm = comm)
	@test plan.global_size == 7
	if rank == 0
		@test plan.local_size == 4
		@test (plan.rstart, plan.rend) == (0, 4)
		# u: [node1_d1, node1_d2, node2_d1, node2_d2], s, v
		@test plan.layout.local_to_global == [1, 2, 5, 6, 3, 4]
		@test plan.layout.owned_mask == [true, true, false, false, true, true]
	else
		@test plan.local_size == 3
		@test (plan.rstart, plan.rend) == (4, 7)
		@test plan.layout.local_to_global == [5, 6, 1, 2, 3, 7]
		@test plan.layout.owned_mask == [true, true, false, false, false, true]
	end
end

@testset "exchange plan communication (rank $rank)" begin
	p = two_node_partition()
	MyState = Compound(:MyState, :u => field((1, 2); field_type = NODAL); mesh = mock_mesh(p), partition_info = p, comm = comm)
	layout_reduced, _ = adapt_layout(Lifter(MyState.size), get_layout(MyState), comm)
	plan = ExchangePlan(layout_reduced; comm = comm)

	scatter_fwd = make_scatter_fwd_set(plan)
	x_owned = rank == 0 ? [10.0] : [20.0]
	u_local = scatter_fwd(x_owned)
	@test u_local == (rank == 0 ? [10.0, 20.0] : [20.0, 10.0])

	# reverse add (assembly): global 1 gets 20 + 20, global 2 gets 40 + 40
	scatter_rev = make_scatter_rev_add(plan, u -> 2.0 .* u)
	@test scatter_rev(u_local) == (rank == 0 ? [40.0] : [80.0])
end

@testset "exchange plan with an incomplete nodal field (rank $rank)" begin
	# 3 nodes; the subset field lives on nodes 1 and 3
	p, subset = rank == 0 ? (PartitionInfo([1, 2], 1), [1]) : (PartitionInfo([2, 3], 2), [2])
	MyState = Compound(:MyState,
		:u => field((1, AUTO); field_type = NODAL),
		:l => field((1, AUTO); field_type = Nodal(node_ids = subset));
		mesh = mock_mesh(p), partition_info = p, comm = comm,
	)
	layout_reduced, _ = adapt_layout(Lifter(MyState.size), get_layout(MyState), comm)
	plan = ExchangePlan(layout_reduced; comm = comm)
	@test plan.global_size == 5
	if rank == 0
		@test plan.local_size == 2
		@test plan.layout.local_to_global == [1, 3, 2]
		@test plan.layout.owned_mask == [true, false, true]
	else
		@test plan.local_size == 3
		@test plan.layout.local_to_global == [3, 4, 5]
		@test plan.layout.owned_mask == [true, true, true]
	end
end

@testset "exchange plan Hessian assembly (rank $rank)" begin
	p = two_node_partition()
	MyState = Compound(:MyState, :u => field((1, 2); field_type = NODAL); mesh = mock_mesh(p), partition_info = p, comm = comm)
	pattern = sparse(ones(Bool, 2, 2))  # one element coupling the two local nodes
	cm = ColoredMatrix(sparse(ones(2, 2)))
	cm = with_data(cm, [1.0, 2.0, 3.0, 4.0] .* (rank == 0 ? 10.0 : 100.0))  # CSR order
	layout_reduced, _ = adapt_layout(Lifter(MyState.size), get_layout(MyState), comm)
	plan = ExchangePlan(layout_reduced, pattern; comm = comm)
	hessian_fn = make_scatter_rev_add(plan, u -> cm; is_hessian = true)
	K = hessian_fn(zeros(2))
	@test K isa ColoredMatrix
	@test plan.owned_nnz == 2
	@test plan.owned_csr[1] == [1, 3]
	@test plan.owned_csr[2] == [1, 2]
	# G11 = 10 + 400, G12 = 20 + 300 (rank 0); G21 = 30 + 200, G22 = 40 + 100 (rank 1)
	@test K.data == (rank == 0 ? [410.0, 320.0] : [230.0, 140.0])
end

@testset "exchange plan with constraints (rank $rank)" begin
	p = two_node_partition()
	MyState = Compound(:MyState, :u => field((1, 2); field_type = NODAL); mesh = mock_mesh(p), partition_info = p, comm = comm)
	node1_local = findall(==(1), p.nodes_local_to_global)  # global node 1 is fixed
	lifter = Lifter(MyState.size, Fixed(node1_local, 0.0))
	layout_reduced, _ = adapt_layout(lifter, get_layout(MyState), comm)
	plan = ExchangePlan(layout_reduced; comm = comm)
	@test plan.global_size == 1
	@test plan.layout.local_to_global == [1]
	if rank == 0
		@test plan.local_size == 0
		@test (plan.rstart, plan.rend) == (0, 0)
	else
		@test plan.local_size == 1
		@test (plan.rstart, plan.rend) == (0, 1)
	end
end

MPI.Finalize()
