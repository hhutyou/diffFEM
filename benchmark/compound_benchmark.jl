# Port of tatva/tests/test_compound_benchmark.py: compound field indexing and updates add
# little overhead compared with manual indexing.
#
#     julia --project=. benchmark/compound_benchmark.jl

using Tatva, Printf, Statistics

function time_median(f; runs = 20)
	f()
	return median([(t0 = time_ns(); f(); (time_ns() - t0) / 1e9) for _ in 1:runs])
end

for n_nodes ∈ (1_000, 10_000, 100_000)
	Plain = Compound(:Plain, :u => field((2, n_nodes)), :v => field((n_nodes,)))
	Stacked = stack_fields(Compound(:Stacked, :u => field((2, n_nodes)), :v => field((n_nodes,))), :u, :v; axis = 1)
	plain = Plain(zeros(Plain.size))
	stacked = Stacked(zeros(Stacked.size))
	node_ids = collect(1:max(1, n_nodes ÷ 10):n_nodes)
	u_value = reshape(collect(1.0:(2n_nodes)), 2, n_nodes)
	v_value = collect(1.0:n_nodes)

	timings = [
		"plain_indices" => time_median(() -> Plain.u[:, node_ids]),
		"stacked_indices" => time_median(() -> Stacked.u[:, node_ids]),
		"plain_set" => time_median(() -> set(set(plain, :u, u_value), :v, v_value).arr),
		"stacked_set" => time_median(() -> set(set(stacked, :u, u_value), :v, v_value).arr),
		"manual_set" => time_median(() -> begin
			a = copy(stacked.arr)
			m = reshape(a, 3, :)
			m[1:2, :] .= u_value
			m[3, :] .= v_value
			a
		end),
		"stacked_view_sum" => time_median(() -> sum(stacked.u)),
	]
	@printf("\nCompound benchmark, n_nodes = %d\n", n_nodes)
	for (name, t) ∈ timings
		@printf("%18s: %.3e s\n", name, t)
	end
	@assert length(Plain.u[:, node_ids]) == 2 * length(node_ids)
end
