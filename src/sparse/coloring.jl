#--------------------------------------------------------------------------------
# Distance-2 graph coloring for sparse Jacobians
#--------------------------------------------------------------------------------
#
# Columns j and k of a Jacobian with sparsity pattern A can share a color (be
# differentiated together with one seed) iff no row contains both, i.e. they are not
# adjacent in the column-intersection graph AᵀA. For the symmetric patterns of FEM
# operators AᵀA = A², which is the "distance-2" graph of tatva's coloring. Colors are
# 1-based.

_pattern_from_csr(indptr::AbstractVector{<:Integer}, indices::AbstractVector{<:Integer}, n::Integer; ncols::Integer = n) =
	copy(transpose(SparseMatrixCSC(Int(ncols), Int(n), Vector{Int}(indptr), Vector{Int}(indices), ones(Bool, length(indices)))))

"""
	distance2_colors(pattern; order=1:n) -> Vector{Int}
	distance2_colors(indptr, indices, n_dofs)

Greedy distance-2 coloring of the columns of the sparsity `pattern` (a sparse matrix, or a
1-based CSR structure): columns sharing a row get different colors. Columns are colored in
`order`, each with the smallest color not used by an already colored neighbour.
"""
distance2_colors(A::SparseMatrixCSC; order = 1:size(A, 2)) = _greedy_distance2(A, order)
distance2_colors(indptr::AbstractVector{<:Integer}, indices::AbstractVector{<:Integer}, n_dofs::Integer) =
	distance2_colors(_pattern_from_csr(indptr, indices, n_dofs))

function _greedy_distance2(A::SparseMatrixCSC, order)
	n = size(A, 2)
	At = copy(transpose(A))  # column r of At lists the columns of row r of A
	colors = zeros(Int, n)
	forbidden = zeros(Int, n + 1)
	stamp = 0
	for j ∈ order
		stamp += 1
		for p ∈ nzrange(A, j)
			r = rowvals(A)[p]
			for q ∈ nzrange(At, r)
				c = colors[rowvals(At)[q]]
				c > 0 && (forbidden[c] = stamp)
			end
		end
		c = 1
		while forbidden[c] == stamp
			c += 1
		end
		colors[j] = c
	end
	return colors
end

# number of distinct distance-2 neighbours of every column (including itself when its column is non-empty)
function _distance2_degrees(A::SparseMatrixCSC)
	n = size(A, 2)
	At = copy(transpose(A))
	seen = zeros(Int, n)
	degrees = zeros(Int, n)
	for j ∈ 1:n
		for p ∈ nzrange(A, j)
			r = rowvals(A)[p]
			for q ∈ nzrange(At, r)
				k = rowvals(At)[q]
				if seen[k] != j
					seen[k] = j
					degrees[j] += 1
				end
			end
		end
	end
	return degrees
end

"""
	largest_degree_first_distance2_colors(pattern)

Greedy distance-2 coloring visiting columns by decreasing distance-2 degree.
"""
largest_degree_first_distance2_colors(A::SparseMatrixCSC) = _greedy_distance2(A, sortperm(_distance2_degrees(A); rev = true))
largest_degree_first_distance2_colors(indptr::AbstractVector{<:Integer}, indices::AbstractVector{<:Integer}, n_dofs::Integer) =
	largest_degree_first_distance2_colors(_pattern_from_csr(indptr, indices, n_dofs))

"""
	smallest_last_distance2_colors(pattern)

Greedy distance-2 coloring in smallest-last order (repeatedly remove a vertex of minimum
degree in the remaining distance-2 graph; color in reverse removal order).
"""
smallest_last_distance2_colors(A::SparseMatrixCSC) = _greedy_distance2(A, _smallest_last_order(A))
smallest_last_distance2_colors(indptr::AbstractVector{<:Integer}, indices::AbstractVector{<:Integer}, n_dofs::Integer) =
	smallest_last_distance2_colors(_pattern_from_csr(indptr, indices, n_dofs))

function _distance2_adjacency(A::SparseMatrixCSC)
	n = size(A, 2)
	At = copy(transpose(A))
	seen = zeros(Int, n)
	adjacency = [Int[] for _ ∈ 1:n]
	for j ∈ 1:n
		for p ∈ nzrange(A, j), q ∈ nzrange(At, rowvals(A)[p])
			k = rowvals(At)[q]
			if k != j && seen[k] != j
				seen[k] = j
				push!(adjacency[j], k)
			end
		end
	end
	return adjacency
end

function _smallest_last_order(A::SparseMatrixCSC)
	adjacency = _distance2_adjacency(A)
	n = length(adjacency)
	degrees = length.(adjacency)
	maxdeg = maximum(degrees; init = 0)
	buckets = [Set{Int}() for _ ∈ 0:maxdeg]
	for (i, d) ∈ enumerate(degrees)
		push!(buckets[d+1], i)
	end
	removed = falses(n)
	order = Int[]
	ptr = 0
	for _ ∈ 1:n
		while isempty(buckets[ptr+1])
			ptr += 1
		end
		v = pop!(buckets[ptr+1])
		removed[v] = true
		push!(order, v)
		for k ∈ adjacency[v]
			removed[k] && continue
			d = degrees[k]
			delete!(buckets[d+1], k)
			degrees[k] = d - 1
			push!(buckets[d], k)
			ptr = min(ptr, d - 1)
		end
	end
	return reverse!(order)
end

"""
	seeds_from_coloring(colors) -> Matrix{Int}

One seed row per color: `seeds[c, j] = 1` if column `j` has color `c`.
"""
function seeds_from_coloring(colors::AbstractVector{<:Integer})
	nc = maximum(colors; init = 0)
	seeds = zeros(Int, nc, length(colors))
	for (j, c) ∈ enumerate(colors)
		seeds[c, j] = 1
	end
	return seeds
end

"""
	distance2_color_and_seeds(pattern) -> (colors, seeds)
	distance2_color_and_seeds(indptr, indices, n_dofs)
"""
distance2_color_and_seeds(A::SparseMatrixCSC) = (c = distance2_colors(A); (c, seeds_from_coloring(c)))
distance2_color_and_seeds(indptr::AbstractVector{<:Integer}, indices::AbstractVector{<:Integer}, n_dofs::Integer) =
	distance2_color_and_seeds(_pattern_from_csr(indptr, indices, n_dofs))
