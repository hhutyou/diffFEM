# End-to-end distributed assembly on a partitioned mesh, checked against the serial
# computation (beyond tatva's unit tests: this exercises extract_local_mesh, the compound
# layout, adapt_layout, the ghost exchange and the reverse-add of gradients and sparse
# Hessians together). Run with 2 ranks:
#     mpiexec -n 2 julia --project test/mpi/test_parallel_assembly.jl

using Test, MPI, Tatva, SparseArrays, LinearAlgebra, Random

MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
@assert MPI.Comm_size(comm) == 2 "this test needs exactly 2 MPI ranks"

function allgather_vec(v::Vector{Int})
	counts = MPI.Allgather(length(v), comm)
	buf = Vector{Int}(undef, sum(counts))
	MPI.Allgatherv!(v, MPI.VBuffer(buf, counts), comm)
	return buf
end

μ, λ = 1.0, 0.7
ψ(∇u) = (F = I + ∇u; J = det(F); μ / 2 * (tr(F' * F) - 2) - μ * log(J) + λ / 2 * log(J)^2)

@testset "distributed gradient and Hessian match the serial ones (rank $rank)" begin
	mesh = unit_square(6, 5)
	nel = size(mesh.elements, 2)
	centroid_x(e) = sum(mesh.coords[1, mesh.elements[:, e]]) / 3
	partition = [centroid_x(e) < 0.5 ? 0 : 1 for e in 1:nel]

	# --- serial reference -------------------------------------------------------------
	op = Operator(mesh, Tri3())
	Global = Compound(:Global, :u => field((2, AUTO)); mesh = mesh)
	left = findall(mesh.coords[1, :] .== 0.0)
	lifter_g = Lifter(Global.size, Fixed(Global.u[:, left], 0.0))
	energy_g(ur) = integrate(op, ψ.(grad(op, Global(lift_from_zeros(lifter_g, ur)).u)))
	u_serial = 0.01 .* randn(MersenneTwister(42), lifter_g.size_reduced)  # same on every rank
	g_serial = jacrev(energy_g)(u_serial)
	cm_g = ColoredMatrix(adapt_sparsity(lifter_g, pattern_from_mesh(mesh, 2)))
	H_serial = sparse(sparse_hessian(energy_g, cm_g)(u_serial))
	serial_pos = zeros(Int, Global.size)  # natural dof → position among the serial free dofs
	serial_pos[lifter_g.free_dofs] = 1:lifter_g.size_reduced

	# --- distributed --------------------------------------------------------------------
	local_mesh, pinfo = extract_local_mesh(mesh, partition, rank)
	op_l = Operator(local_mesh, Tri3())
	Local = Compound(:Local, :u => field((2, AUTO)); mesh = local_mesh, partition_info = pinfo, comm = comm)
	layout = get_layout(Local)
	left_l = findall(local_mesh.coords[1, :] .== 0.0)
	lifter = Lifter(Local.size, Fixed(Local.u[:, left_l], 0.0))
	layout_r, lifter_r = adapt_layout(lifter, layout, comm)
	local_pattern = adapt_sparsity(lifter_r, pattern_from_mesh(local_mesh, 2))
	plan = ExchangePlan(layout_r, local_pattern; comm = comm)
	@test plan.global_size == lifter_g.size_reduced

	# natural (= serial flat) index of every free local dof
	natural = layout.natural_l2g[lifter_r.free_dofs]
	owned = findall(layout_r.owned_mask)
	pos_owned = layout_r.local_to_global[owned] .- plan.rstart

	x_owned = zeros(plan.local_size)
	x_owned[pos_owned] = u_serial[serial_pos[natural[owned]]]
	u_local = make_scatter_fwd_set(plan)(x_owned)
	@test u_local ≈ u_serial[serial_pos[natural]]  # ghosts received from their owners

	energy_l(ur) = integrate(op_l, ψ.(grad(op_l, Local(lift_from_zeros(lifter_r, ur)).u)))
	g_owned = make_scatter_rev_add(plan, jacrev(energy_l))(u_local)
	@test g_owned[pos_owned] ≈ g_serial[serial_pos[natural[owned]]] atol = 1e-12

	cm_l = ColoredMatrix(local_pattern)
	H_owned = make_scatter_rev_add(plan, sparse_hessian(energy_l, cm_l); is_hessian = true)(u_local)
	# owned rows also couple to dofs held only by the other rank: gather the global map
	solver_to_serial = Dict(zip(allgather_vec(layout_r.local_to_global), allgather_vec(serial_pos[natural])))
	owned_rows_serial = Dict(layout_r.local_to_global[k] - plan.rstart => serial_pos[natural[k]] for k in owned)
	ptr, cols = plan.owned_csr
	max_err = 0.0
	n_checked = 0
	for i ∈ 1:plan.local_size, k ∈ ptr[i]:(ptr[i+1]-1)
		r, c = owned_rows_serial[i], solver_to_serial[cols[k]]
		max_err = max(max_err, abs(H_owned.data[k] - H_serial[r, c]))
		n_checked += 1
	end
	@test max_err < 1e-12
	# every nonzero of the owned serial rows is present
	@test n_checked == sum(nnz(H_serial[r:r, :]) for r in values(owned_rows_serial))
	@test MPI.Allreduce(n_checked, +, comm) == nnz(H_serial)
end

MPI.Finalize()
