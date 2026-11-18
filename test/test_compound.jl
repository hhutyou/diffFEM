# Port of tatva/tests/test_compound.py, plus tests of the Julia API (layouts as objects,
# inheritance, macro, patterns).
#
# Shapes follow the Julia convention (reversed w.r.t. tatva): tatva's `field((2, 3))` is
# `field((3, 2))` here, with the same flat layout; indices are 1-based.

using Test, Tatva, SparseArrays, ForwardDiff

const SimpleState = Compound(:SimpleState, :u => field((3, 2)), :phi => field((2,)))
const StackedState = stack_fields(
	Compound(:StackedState, :u => field((2, 2)), :v => field((2, 2)), :w => field((2,))),
	:u, :v; axis = 1,
)

@testset "compound size matches the flat array: $(layout.name)" for layout in (SimpleState, StackedState)
	state = layout()
	@test size(state.arr) == (layout.size,)
	@test all(iszero, state.arr)
end

@testset "field access and assignment" begin
	state = SimpleState()
	@test SimpleState.size == 8
	@test state.u == zeros(3, 2)
	@test state.phi == zeros(2)

	u_val = reshape(collect(0.0:5.0), 3, 2)
	phi_val = [10.0, 20.0]
	state = set(state, :u, u_val)
	state = set(state, :phi, phi_val)
	@test state.u == u_val
	@test state.phi == phi_val
	@test state.arr[1:6] == vec(u_val)
	@test state.arr[7:8] == phi_val

	@test length(state) == 2
	@test [size(c) for c in state] == [(3, 2), (2,)]
	# tatva's `state.at("u").set(value)`
	@test state.at(:u).set(zeros(3, 2)).u == zeros(3, 2)
	@test state.u == u_val  # functional update: the original is unchanged
end

@testset "index helpers" begin
	@test SimpleState.u[2] == [4, 5, 6]     # tatva: SimpleState.u[1] == [3, 4, 5]
	@test SimpleState.u[:, 2] == [4, 5, 6]
	@test SimpleState.u[2, :] == [2, 5]     # tatva: SimpleState.u[:, 1] == [1, 4]
	@test SimpleState.phi[2] == [8]         # tatva: SimpleState.phi[1] == [7]
	@test SimpleState.u[2, [1, end]] == [2, 5]
	@test SimpleState.u[end, end] == [6]
	@test indices(SimpleState, :u, 1, 1) == [1]
	@test_throws BoundsError SimpleState.u[4, 1]
end

@testset "state round trip and addition" begin
	arr_a = collect(0.0:(SimpleState.size-1))
	arr_b = 2 .* arr_a
	state_a = SimpleState(arr_a)
	state_b = SimpleState(arr_b)
	@test flatten(state_a) === arr_a
	rebuilt = SimpleState(flatten(state_a))
	@test rebuilt.u == state_a.u && rebuilt.phi == state_a.phi
	summed = state_a + state_b
	@test summed.arr == arr_a + arr_b
	@test summed.u == state_a.u + state_b.u
	@test summed.phi == state_a.phi + state_b.phi
	@test_throws DimensionMismatch SimpleState(zeros(3))
end

@testset "stack_fields access and indices" begin
	state = StackedState(collect(0.0:(StackedState.size-1)))
	@test state.u == [0.0 4.0; 1.0 5.0]   # tatva: [[0, 1], [4, 5]]
	@test state.v == [2.0 6.0; 3.0 7.0]   # tatva: [[2, 3], [6, 7]]
	@test state.w == [8.0, 9.0]
	@test StackedState.u[2] == [5, 6]     # tatva: StackedState.u[1] == [4, 5]
	@test StackedState.v[1] == [3, 4]     # tatva: StackedState.v[0] == [2, 3]
	# setting a stacked field writes through the strided view
	s2 = set(state, :v, [-1.0 -3.0; -2.0 -4.0])
	@test s2.arr == [0, 1, -1, -2, 4, 5, -3, -4, 8, 9]
	@test_throws CompoundError stack_fields(SimpleState, :nope)
	@test_throws CompoundError stack_fields(StackedState, :u)  # already stacked
end

@testset "auto sizing of nodal fields" begin
	mesh = Mesh(zeros(2, 10))
	MyState = Compound(:MyState,
		:param1 => field((5,)),
		:u => field((3, AUTO)),
		:phi => field((AUTO,)),
		:param2 => field((2,));
		mesh = mesh,
	)
	state = MyState()
	@test collect(keys(MyState.fields)) == [:param1, :u, :phi, :param2]  # declaration order
	p1, u, phi, p2 = state
	@test size(p1) == (5,) && size(u) == (3, 10) && size(phi) == (10,) && size(p2) == (2,)
	@test MyState.u.shape == (3, 10)
	@test MyState.phi.shape == (10,)
	@test MyState.size == 47
	@test length(state.arr) == 47
	@test MyState.u.field_type == NODAL && MyState.param1.field_type == LOCAL

	# memory layout: the nodal block first (u and phi stacked node by node), then the rest
	@test indices(MyState.u) == [(i - 1) * 4 + j for i in 1:10 for j in 1:3]
	@test indices(MyState.phi) == [(i - 1) * 4 + 4 for i in 1:10]
	@test indices(MyState.param1) == 41:45
	@test indices(MyState.param2) == 46:47
	@test_throws CompoundError Compound(:NoMesh, :u => field((2, AUTO)))
end

@testset "default factories, keyword initialisation, element types" begin
	S = Compound(:S, :u => field((2,); default_factory = () -> [1.0, 2.0]), :v => field((3,)))
	@test S().u == [1.0, 2.0]
	@test S(v = [7.0, 8.0, 9.0]).v == [7.0, 8.0, 9.0]
	@test S(v = [7.0, 8.0, 9.0]).u == [1.0, 2.0]
	@test S(u = 4.0).u == [4.0, 4.0]
	d = ForwardDiff.Dual(1.0, 1.0)
	@test eltype(set(S(), :u, [d, d]).arr) == typeof(d)
end

@testset "differentiating through compound views" begin
	S = Compound(:S, :u => field((2, 3)), :phi => field((3,)))
	E(arr) = (s = S(arr); sum(abs2, s.u) + sum(s.phi .* s.u[1, :]))
	x = collect(1.0:S.size)
	g = ForwardDiff.gradient(E, x)
	s = S(x)
	@test S(g).u ≈ 2 .* s.u .+ [s.phi'; zeros(1, 3)]
	@test S(g).phi ≈ s.u[1, :]
end

@testset "inheritance and the @compound macro" begin
	Parent = Compound(:Parent, :a => field((2,)), :b => field((3,)))
	Child = Tatva.extend(Parent, :c => field((4,)); name = :Child)
	@test collect(keys(Child.fields)) == [:a, :b, :c]
	@test Child.size == 9
	@test indices(Child.c) == 6:9
	@test indices(Child.a) == indices(Parent.a)

	n = 3
	@compound Macro begin
		x = field((2,))
		y = field((n,); default_factory = () -> ones(n))
	end
	@test Macro.name == :Macro
	@test Macro().y == ones(3)
	@compound Derived parent = Macro begin
		z = field((1,))
	end
	@test collect(keys(Derived.fields)) == [:x, :y, :z]
end

@testset "invalid definitions" begin
	@test_throws CompoundError Compound(:Bad, :arr => field((2,)))
	@test_throws CompoundError Compound(:Bad, :size => field((2,)))
	@test_throws CompoundError Compound(:Bad, :u => field((2,)), :u => field((3,)))
end

@testset "nodal fields on a node subset and unstacked nodal fields" begin
	mesh = unit_square(2, 2)  # 9 nodes
	S = Compound(:S,
		:u => field((2, AUTO)),
		:lm => field((1, AUTO); field_type = Nodal(node_ids = [1, 4, 7])),
		:p => field((AUTO,); field_type = Nodal(stack = false));
		mesh = mesh,
	)
	@test S.lm.shape == (1, 3)
	@test !Tatva.is_stacked(S.u)  # u is the only stackable full nodal field
	@test S.size == 18 + 3 + 9
end

@testset "pattern_from_compound" begin
	mesh = unit_square(3, 2)
	# two stacked nodal fields with 2 + 1 dofs per node = an interleaved 3-dof field
	S = Compound(:S, :u => field((2, AUTO)), :phi => field((AUTO,)); mesh = mesh)
	@test pattern_from_compound(S) == pattern_from_mesh(mesh, 3)

	# a local field only gets diagonal entries (with a warning)
	S2 = Compound(:S2, :u => field((2, AUTO)), :phi => field((AUTO,)), :g => field((2,)); mesh = mesh)
	pat = @test_logs (:warn, r"Custom space detected for field 'g'") pattern_from_compound(S2)
	n3 = 3 * num_nodes(mesh)
	@test pat[1:n3, 1:n3] == pattern_from_mesh(mesh, 3)
	@test pat[(n3+1):end, (n3+1):end] == sparse(trues(2, 2) .& [true false; false true])
	@test nnz(pat[1:n3, (n3+1):end]) == 0

	blocks = @test_logs (:warn,) pattern_from_compound(S2; block_wise = true)
	@test size(blocks) == (2, 2)
	@test blocks[1, 1] == pattern_from_mesh(mesh, 3)
	@test nnz(blocks[1, 2]) == 0 && nnz(blocks[2, 1]) == 0

	# a field on a subset of nodes couples within the elements touching those nodes
	S3 = Compound(:S3, :u => field((AUTO,)), :lm => field((1, AUTO); field_type = Nodal(node_ids = [1])); mesh = mesh)
	p3 = pattern_from_compound(S3)
	lm_dof = indices(S3.lm)[1]
	touching = unique(vec(mesh.elements[:, findall(e -> 1 in mesh.elements[:, e], 1:size(mesh.elements, 2))]))
	@test sort(findall(p3[lm_dof, :])) == sort(vcat(indices(S3.u)[touching], lm_dof))
end
