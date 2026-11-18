# A structured tetrahedral mesh of [0, L₁] × [0, L₂] × [0, L₃]: nx × ny × nz cells of 6 tetrahedra.
function box_mesh(nx, ny, nz; L = (4.0, 1.0, 1.0))
	xs, ys, zs = (range(0.0, L[k]; length = n + 1) for (k, n) in enumerate((nx, ny, nz)))
	sy, sz = nx + 1, (nx + 1) * (ny + 1)
	coords = zeros(3, sz * (nz + 1))
	for k ∈ 0:nz, j ∈ 0:ny, i ∈ 0:nx
		coords[:, i+j*sy+k*sz+1] .= (xs[i+1], ys[j+1], zs[k+1])
	end
	tets = NTuple{4, Int}[]
	for k ∈ 0:(nz-1), j ∈ 0:(ny-1), i ∈ 0:(nx-1)
		n0 = i + j * sy + k * sz
		n1, n2, n4 = n0 + 1, n0 + sy, n0 + sz
		n3, n5, n6 = n2 + 1, n4 + 1, n4 + sy
		n7 = n6 + 1
		push!(tets, (n0, n1, n3, n7), (n0, n1, n7, n5), (n0, n5, n7, n4), (n0, n3, n2, n7), (n0, n2, n6, n7), (n0, n6, n4, n7))
	end
	return Mesh(coords, [t[a] + 1 for a in 1:4, t in tets])
end
