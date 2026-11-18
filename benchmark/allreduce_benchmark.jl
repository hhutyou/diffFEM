# Port of tatva/tests/test_allreduce_benchmark.py: allreduce-based parallel assembly of the
# gradient and sparse Hessian of a 2D linear-elastic problem (elements split across ranks,
# replicated dof vector). Reports the slowest rank.
#
#     mpiexec -n 4 julia --project=. benchmark/allreduce_benchmark.jl

using MPI, Tatva, LinearAlgebra, Printf

MPI.Init()
comm = MPI.COMM_WORLD
rank, nranks = MPI.Comm_rank(comm), MPI.Comm_size(comm)

ψ(∇u) = (ε = (∇u + ∇u') / 2; 0.5 * dot(2.0 * ε + 0.3 * tr(ε) * I, ε))

for n ∈ (20, 40, 80)
	mesh = unit_square(n, n)
	nel = size(mesh.elements, 2)
	chunk = cld(nel, nranks)
	local_elements = mesh.elements[:, (rank*chunk+1):min((rank + 1) * chunk, nel)]
	local_op = Operator(Mesh(mesh.coords, local_elements), Tri3())  # all nodes, own elements

	n_dofs = 2 * num_nodes(mesh)
	bottom = findall(mesh.coords[2, :] .== 0.0)
	lifter = Lifter(n_dofs, Fixed(vcat(2 .* bottom .- 1, 2 .* bottom), 0.0))
	pattern = adapt_sparsity(lifter, pattern_from_mesh(mesh, 2))

	# local contributions with the element-wise assembler, reduced to the free dofs
	asm = ElementAssembler(local_op, 2)
	density(ξ, Ue, Xe) = ψ(grad(local_op.element, ξ, Ue, Xe))
	cm = ColoredMatrix(pattern)
	rows, cols = Tatva.csr_rows(cm), cm.indices
	local_grad(u_free) = reduce_adjoint(lifter, assemble_gradient(asm, density, reshape(lift_from_zeros(lifter, u_free), 2, :), mesh.coords))
	function local_hessian(u_free)
		K = reduce_hessian(lifter, assemble_hessian(asm, density, reshape(lift_from_zeros(lifter, u_free), 2, :), mesh.coords))
		return with_data(cm, [K[i, j] for (i, j) in zip(rows, cols)])  # values on the global pattern (CSR order)
	end

	plan = AllreducePlan(lifter.size_reduced, pattern; comm = comm)
	allgather = make_allgather(plan)
	grad_fn = make_allreduce_owned(plan, local_grad)
	hess_fn = make_allreduce_owned(plan, local_hessian; is_hessian = true)

	u_full = allgather(zeros(plan.local_size))
	grad_fn(u_full)
	hess_fn(u_full)
	MPI.Barrier(comm)
	t_grad = @elapsed for _ ∈ 1:5
		grad_fn(u_full)
	end
	MPI.Barrier(comm)
	t_hess = @elapsed for _ ∈ 1:5
		hess_fn(u_full)
	end
	t_grad = MPI.Reduce(t_grad / 5, max, comm; root = 0)
	t_hess = MPI.Reduce(t_hess / 5, max, comm; root = 0)
	rank == 0 && @printf("P=%d mesh %3d×%-3d (%6d elements, %7d free dofs): grad %.4f s, hessian %.4f s\n",
		nranks, n, n, nel, lifter.size_reduced, t_grad, t_hess)
end

MPI.Finalize()
