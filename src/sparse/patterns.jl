#--------------------------------------------------------------------------------
# Sparsity patterns from the mesh / compound structure
#--------------------------------------------------------------------------------
#
# Patterns are `SparseMatrixCSC{Bool,Int}` with a `true` per structural nonzero.

function _pattern_from_element_dofs(element_dofs::AbstractMatrix{<:Integer}, n_rows::Integer, n_cols::Integer = n_rows)
	m, nel = size(element_dofs)
	rows = Int[]
	cols = Int[]
	sizehint!(rows, m * m * nel)
	sizehint!(cols, m * m * nel)
	for e ∈ 1:nel, j ∈ 1:m
		dj = element_dofs[j, e]
		dj > 0 || continue  # 0 marks a missing dof (incomplete nodal fields)
		for i ∈ 1:m
			di = element_dofs[i, e]
			di > 0 || continue
			push!(rows, di)
			push!(cols, dj)
		end
	end
	return sparse(rows, cols, fill(true, length(rows)), Int(n_rows), Int(n_cols), |)
end

# dofs of every element for `n_dofs_per_node` interleaved dofs per node: (npe*d, n_elements)
function _element_dofs(elements::AbstractMatrix{<:Integer}, n_dofs_per_node::Integer)
	d = Int(n_dofs_per_node)
	npe, nel = size(elements)
	dofs = Matrix{Int}(undef, npe * d, nel)
	for e ∈ 1:nel, a ∈ 1:npe, c ∈ 1:d
		dofs[(a-1)*d+c, e] = (elements[a, e] - 1) * d + c
	end
	return dofs
end

"""
	pattern_from_mesh(mesh, n_dofs_per_node) -> SparseMatrixCSC{Bool}

Sparsity pattern of an operator coupling all dofs of every element, with
`n_dofs_per_node` interleaved dofs per node (dof `(node - 1) * d + c`), as for a field
`reshape(u, d, :)`.
"""
function pattern_from_mesh(mesh::Mesh, n_dofs_per_node::Integer)
	n = n_dofs_per_node * num_nodes(mesh)
	return _pattern_from_element_dofs(_element_dofs(mesh.elements, n_dofs_per_node), n)
end

_pattern_from_elements(elements::AbstractMatrix{<:Integer}, n_dofs_per_node::Integer, n::Integer) =
	_pattern_from_element_dofs(_element_dofs(elements, n_dofs_per_node), n)

"""
	pattern_from_compound(layout; block_wise=false)

Sparsity pattern of the operator of a [`Compound`](@ref) with an attached mesh: nodal fields
are fully coupled within elements (with each other too), other fields (local, shared) only
get diagonal entries (with a warning — provide your own pattern if they couple). With
`block_wise=true` returns the pattern split into blocks, one per contiguous field block
(stacked fields form one block), as a matrix of sparse matrices.
"""
function pattern_from_compound(layout::CompoundLayout; block_wise::Bool = false)
	mesh = layout.mesh
	mesh === nothing && throw(CompoundError("Mesh must be set on the compound to create a sparsity pattern."))
	n_nodes = num_nodes(mesh)
	elements = mesh.elements
	nel = size(elements, 2)
	coupled = Matrix{Int}[]
	diagonal = Int[]
	for (name, f) ∈ pairs(layout.fields)
		ft = f.field_type
		if ft isa Nodal
			idx = indices(f)
			n_items = ft.node_ids === nothing ? n_nodes : length(ft.node_ids)
			n_items > 0 || continue
			per_item = length(idx) ÷ n_items
			node_dofs = zeros(Int, per_item, n_nodes)
			if ft.node_ids === nothing
				node_dofs .= reshape(idx, per_item, n_nodes)
			else
				item_dofs = reshape(idx, per_item, n_items)
				for (k, node) ∈ enumerate(ft.node_ids)
					1 <= node <= n_nodes && (node_dofs[:, node] .= item_dofs[:, k])
				end
			end
			push!(coupled, reshape(node_dofs[:, vec(elements)], per_item * size(elements, 1), nel))
		else
			@warn "Custom space detected for field '$name'. Only diagonal entries added to the sparsity pattern. " *
				  "Please provide your own sparsity pattern if cross-coupling is required."
			append!(diagonal, indices(f))
		end
	end
	n = layout.size
	full = isempty(coupled) ? spzeros(Bool, n, n) : _pattern_from_element_dofs(reduce(vcat, coupled), n)
	if !isempty(diagonal)
		full = full .| sparse(diagonal, diagonal, fill(true, length(diagonal)), n, n)
	end
	full = _binarize(full)
	block_wise || return full
	roots = sort!(unique([f.root for f in values(layout.fields)]); by = first)
	return [full[r1, r2] for r1 in roots, r2 in roots]
end
