#--------------------------------------------------------------------------------
# Finite elements
#--------------------------------------------------------------------------------
#
# Element-level conventions:
# - ξ is an `SVector{D}` of reference coordinates;
# - nodal values `V` are a `StaticArray` with the node axis LAST: `SVector{N}` for a scalar
#   field, `SMatrix{M,N}` for a vector field, `SArray{Tuple{A,B,N}}` for a tensor field;
# - nodal coordinates `X` are an `SMatrix{DX,N}` (one column per node);
# - shape-function derivatives are `SMatrix{D,N}` with `dNdr[i, n] = ∂Nₙ/∂ξᵢ`;
# - the Jacobian is `J = dNdr * X'`, i.e. `J[i, j] = ∂xⱼ/∂ξᵢ` (as in tatva);
# - gradients append a trailing spatial axis to the value: scalar → `SVector{DX}`,
#   `SVector{M}` → `SMatrix{M,DX}` (`∇u[i, j] = ∂uᵢ/∂xⱼ`), `SMatrix{A,B}` → `SArray{A,B,DX}`.
#
# Plain arrays are accepted everywhere for convenience and converted to static arrays.

"""
	Element{D,N,NQ}

Abstract supertype of all finite elements: `D` is the reference dimension, `N` the number
of nodes and `NQ` the number of quadrature points. Concrete elements store their
quadrature rule (`quad_points::SVector{NQ,SVector{D,Float64}}`, `quad_weights`) and are
compared by value.

Subtypes implement `reference_nodes`, `default_quadrature`, `shape_function` and
`shape_function_derivative`; everything else (`get_jacobian`, `interpolate`, `gradient`,
`hessian`, `get_local_values`) is generic.
"""
abstract type Element{D, N, NQ} end

reference_dim(::Element{D}) where {D} = D
nodes_per_element(::Element{D, N}) where {D, N} = N
num_quad_points(::Element{D, N, NQ}) where {D, N, NQ} = NQ
reference_dim(::Type{<:Element{D}}) where {D} = D
nodes_per_element(::Type{<:Element{D, N}}) where {D, N} = N

Base.:(==)(a::Element, b::Element) =
	typeof(a).name === typeof(b).name && a.quad_points == b.quad_points && a.quad_weights == b.quad_weights
Base.hash(el::Element, h::UInt) = hash(el.quad_weights, hash(el.quad_points, hash(nameof(typeof(el)), h)))
Base.show(io::IO, el::Element) = print(io, nameof(typeof(el)), "(nq=", num_quad_points(el), ")")

# shared constructor logic: from default or user-supplied quadrature
function _quadrature(::Val{D}, points, weights) where {D}
	pts = _points_vector(Val(D), points)
	NQ = length(pts)
	length(weights) == NQ || throw(DimensionMismatch("got $NQ quadrature points but $(length(weights)) weights"))
	return SVector{NQ}(pts), SVector{NQ, Float64}(weights)
end
_points_vector(::Val{D}, points::AbstractVector{<:AbstractVector}) where {D} = [SVector{D, Float64}(p) for p in points]
_points_vector(::Val{D}, points::AbstractMatrix) where {D} = begin
	size(points, 1) == D || throw(DimensionMismatch("quadrature points must be a ($D, n_q) matrix, got $(size(points))"))
	[SVector{D, Float64}(view(points, :, q)) for q in axes(points, 2)]
end

for (E, D, N, doc) in (
	(:Line2, 1, 2, "A 2-node linear interval element."),
	(:Line3, 1, 3, "A 3-node quadratic line element (nodes: end, end, midpoint)."),
	(:Tri3, 2, 3, "A 3-node linear triangular element."),
	(:Tri6, 2, 6, "A 6-node quadratic triangle; vertices then edge midpoints 0:(0,0) 1:(1,0) 2:(0,1) 3:(½,0) 4:(½,½) 5:(0,½)."),
	(:Quad4, 2, 4, "A 4-node bilinear quadrilateral element."),
	(:Quad8, 2, 8, "An 8-node serendipity quadrilateral element (corners then edge midpoints)."),
	(:Tetrahedron4, 3, 4, "A 4-node linear tetrahedral element."),
	(:Hexahedron8, 3, 8, "An 8-node trilinear hexahedral element."),
)
	@eval begin
		"""
			$($(string(E)))()
			$($(string(E)))(quad_points, quad_weights)

		$($doc) Optionally takes a custom quadrature rule: `quad_points` as a vector of
		points or a `(D, n_q)` matrix, and the matching weights.
		"""
		struct $E{NQ} <: Element{$D, $N, NQ}
			quad_points::SVector{NQ, SVector{$D, Float64}}
			quad_weights::SVector{NQ, Float64}
		end
		function $E(quad_points, quad_weights)
			pts, wts = _quadrature(Val($D), quad_points, quad_weights)
			return $E{length(pts)}(pts, wts)
		end
		$E() = $E(default_quadrature($E)...)
	end
end

#--------------------------------------------------------------------------------
# reference nodes, quadrature rules, shape functions
#--------------------------------------------------------------------------------

"""
	reference_nodes(el) -> SMatrix{D,N}

Nodal positions in natural coordinates, one column per node.
"""
function reference_nodes end

"""
	default_quadrature(ElementType) -> (points, weights)
"""
function default_quadrature end

"""
	shape_function(el, ξ) -> SVector{N}
"""
function shape_function end

"""
	shape_function_derivative(el, ξ) -> SMatrix{D,N}

`dNdr[i, n] = ∂Nₙ/∂ξᵢ`.
"""
function shape_function_derivative end

const _g2 = 1 / sqrt(3.0)
const _g3 = sqrt(3.0 / 5.0)

# --- Line2 ---
reference_nodes(::Line2) = SMatrix{1, 2}(-1.0, 1.0)
default_quadrature(::Type{<:Line2}) = ([SVector(0.0)], [2.0])
shape_function(::Line2, ξ) = (r = ξ[1]; SVector(0.5 * (1 - r), 0.5 * (1 + r)))
shape_function_derivative(::Line2, ξ) = SMatrix{1, 2}(-0.5 + 0 * ξ[1], 0.5 + 0 * ξ[1])

# --- Line3 ---
reference_nodes(::Line3) = SMatrix{1, 3}(-1.0, 1.0, 0.0)
default_quadrature(::Type{<:Line3}) = ([SVector(-_g3), SVector(0.0), SVector(_g3)], [5 / 9, 8 / 9, 5 / 9])
shape_function(::Line3, ξ) = (r = ξ[1]; SVector(0.5 * r * (r - 1), 0.5 * r * (r + 1), 1 - r^2))
shape_function_derivative(::Line3, ξ) = (r = ξ[1]; SMatrix{1, 3}(r - 0.5, r + 0.5, -2r))

# --- Tri3 ---
reference_nodes(::Tri3) = SMatrix{2, 3}(0.0, 0.0, 1.0, 0.0, 0.0, 1.0)
default_quadrature(::Type{<:Tri3}) = ([SVector(1 / 3, 1 / 3)], [1 / 2])
shape_function(::Tri3, ξ) = SVector(1 - ξ[1] - ξ[2], ξ[1], ξ[2])
shape_function_derivative(::Tri3, ξ) = (z = 0 * ξ[1]; SMatrix{2, 3}(-1 + z, -1 + z, 1 + z, z, z, 1 + z))

# --- Tri6 ---
reference_nodes(::Tri6) = SMatrix{2, 6}(0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.5, 0.5, 0.0, 0.5)
default_quadrature(::Type{<:Tri6}) =
	([SVector(1 / 6, 1 / 6), SVector(2 / 3, 1 / 6), SVector(1 / 6, 2 / 3)], [1 / 6, 1 / 6, 1 / 6])
function shape_function(::Tri6, ξ)
	r, s = ξ[1], ξ[2]
	t = 1 - r - s
	return SVector(t * (2t - 1), r * (2r - 1), s * (2s - 1), 4r * t, 4r * s, 4s * t)
end
function shape_function_derivative(::Tri6, ξ)
	r, s = ξ[1], ξ[2]
	t = 1 - r - s
	z = 0 * r
	# rows: d/dr, d/ds; listed column-major (node by node)
	return SMatrix{2, 6}(
		-(4t - 1), -(4t - 1),
		4r - 1, z,
		z, 4s - 1,
		4 * (t - r), -4r,
		4s, 4r,
		-4s, 4 * (t - s),
	)
end

# --- Quad4 ---
reference_nodes(::Quad4) = SMatrix{2, 4}(-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0)
default_quadrature(::Type{<:Quad4}) =
	([SVector(-_g2, -_g2), SVector(_g2, -_g2), SVector(-_g2, _g2), SVector(_g2, _g2)], [1.0, 1.0, 1.0, 1.0])
function shape_function(::Quad4, ξ)
	r, s = ξ[1], ξ[2]
	return 0.25 * SVector((1 - r) * (1 - s), (1 + r) * (1 - s), (1 + r) * (1 + s), (1 - r) * (1 + s))
end
function shape_function_derivative(::Quad4, ξ)
	r, s = ξ[1], ξ[2]
	return 0.25 * SMatrix{2, 4}(
		-(1 - s), -(1 - r),
		(1 - s), -(1 + r),
		(1 + s), (1 + r),
		-(1 + s), (1 - r),
	)
end

# --- Quad8 ---
reference_nodes(::Quad8) = SMatrix{2, 8}(
	-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0,
	0.0, -1.0, 1.0, 0.0, 0.0, 1.0, -1.0, 0.0,
)
function default_quadrature(::Type{<:Quad8})
	x = (-_g3, 0.0, _g3)
	w = (5 / 9, 8 / 9, 5 / 9)
	# r varies fastest, as numpy's meshgrid(..., indexing="xy").ravel()
	pts = [SVector(x[j], x[i]) for i in 1:3 for j in 1:3]
	wts = [w[i] * w[j] for i in 1:3 for j in 1:3]
	return pts, wts
end
function shape_function(::Quad8, ξ)
	r, s = ξ[1], ξ[2]
	return SVector(
		0.25 * (1 - r) * (1 - s) * (-r - s - 1),
		0.25 * (1 + r) * (1 - s) * (r - s - 1),
		0.25 * (1 + r) * (1 + s) * (r + s - 1),
		0.25 * (1 - r) * (1 + s) * (-r + s - 1),
		0.5 * (1 - r^2) * (1 - s),
		0.5 * (1 + r) * (1 - s^2),
		0.5 * (1 - r^2) * (1 + s),
		0.5 * (1 - r) * (1 - s^2),
	)
end
function shape_function_derivative(::Quad8, ξ)
	r, s = ξ[1], ξ[2]
	return SMatrix{2, 8}(
		0.25 * (-2r - s) * (s - 1), 0.25 * (-r - 2s) * (r - 1),
		0.25 * (-2r + s) * (s - 1), 0.25 * (-r + 2s) * (r + 1),
		0.25 * (2r + s) * (s + 1), 0.25 * (r + 1) * (r + 2s),
		0.25 * (2r - s) * (s + 1), 0.25 * (r - 1) * (r - 2s),
		r * (s - 1), 0.5 * r^2 - 0.5,
		0.5 - 0.5 * s^2, -s * (r + 1),
		-r * (s + 1), 0.5 - 0.5 * r^2,
		0.5 * s^2 - 0.5, s * (r - 1),
	)
end

# --- Tetrahedron4 ---
reference_nodes(::Tetrahedron4) = SMatrix{3, 4}(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
default_quadrature(::Type{<:Tetrahedron4}) = ([SVector(1 / 4, 1 / 4, 1 / 4)], [1 / 6])
shape_function(::Tetrahedron4, ξ) = SVector(1 - ξ[1] - ξ[2] - ξ[3], ξ[1], ξ[2], ξ[3])
shape_function_derivative(::Tetrahedron4, ξ) =
	(z = 0 * ξ[1]; SMatrix{3, 4}(-1 + z, -1 + z, -1 + z, 1 + z, z, z, z, 1 + z, z, z, z, 1 + z))

# --- Hexahedron8 ---
const _hex_signs = ((-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1))
reference_nodes(::Hexahedron8) = SMatrix{3, 8}(Float64.(Iterators.flatten(_hex_signs))...)
default_quadrature(::Type{<:Hexahedron8}) = ([SVector{3}(_g2 .* s) for s in _hex_signs], ones(8))
function shape_function(::Hexahedron8, ξ)
	x, y, z = ξ[1], ξ[2], ξ[3]
	return SVector(ntuple(n -> (1 + _hex_signs[n][1] * x) * (1 + _hex_signs[n][2] * y) * (1 + _hex_signs[n][3] * z) / 8, Val(8)))
end
function shape_function_derivative(::Hexahedron8, ξ)
	x, y, z = ξ[1], ξ[2], ξ[3]
	vals = ntuple(Val(24)) do k
		n = (k - 1) ÷ 3 + 1
		i = (k - 1) % 3 + 1
		sx, sy, sz = _hex_signs[n]
		i == 1 ? sx * (1 + sy * y) * (1 + sz * z) / 8 :
		i == 2 ? sy * (1 + sx * x) * (1 + sz * z) / 8 :
		sz * (1 + sx * x) * (1 + sy * y) / 8
	end
	return SMatrix{3, 8}(vals)
end

#--------------------------------------------------------------------------------
# static-array helpers for nodal values (node axis last)
#--------------------------------------------------------------------------------

# Σₙ w[n] V[..., n]
@inline _contract_nodes(V::SVector{N}, w::SVector{N}) where {N} = sum(V .* w)
@inline _contract_nodes(V::SMatrix{M, N}, w::SVector{N}) where {M, N} = V * w
@inline function _contract_nodes(V::SArray{Tuple{A, B, N}}, w::SVector{N}) where {A, B, N}
	r = SMatrix{A * B, N}(Tuple(V)) * w
	return SMatrix{A, B}(Tuple(r))
end

# G[..., j] = Σₙ V[..., n] dNdX[j, n]
@inline _grad_nodes(V::SVector{N}, dNdX::SMatrix{DX, N}) where {N, DX} = dNdX * V
@inline _grad_nodes(V::SMatrix{M, N}, dNdX::SMatrix{DX, N}) where {M, N, DX} = V * transpose(dNdX)
@inline function _grad_nodes(V::SArray{Tuple{A, B, N}}, dNdX::SMatrix{DX, N}) where {A, B, N, DX}
	r = SMatrix{A * B, N}(Tuple(V)) * transpose(dNdX)
	return SArray{Tuple{A, B, DX}}(Tuple(r))
end

# flatten a point value to a static vector, and back
@inline _flatvec(x::Number) = SVector(x)
@inline _flatvec(x::StaticArray) = SVector{length(x)}(Tuple(x))

# Conversions used when element functions are called with plain arrays
_static_ref(::Element{D}, ξ::StaticVector{D}) where {D} = ξ
_static_ref(::Element{D}, ξ) where {D} = SVector{D}(ξ...)
_static_coords(::Element{D, N}, X::StaticMatrix{DX, N}) where {D, N, DX} = X
_static_coords(::Element{D, N}, X::AbstractMatrix) where {D, N} = SMatrix{size(X, 1), N}(X)
_static_nodal(::Element{D, N}, V::StaticArray) where {D, N} = V
function _static_nodal(::Element{D, N}, V::AbstractArray{<:Number}) where {D, N}
	size(V, ndims(V)) == N || throw(DimensionMismatch("nodal values must have the node axis last with $N nodes, got size $(size(V))"))
	return SArray{Tuple{size(V)...}}(V)
end
function _static_nodal(::Element{D, N}, V::AbstractVector{<:StaticArray}) where {D, N}
	length(V) == N || throw(DimensionMismatch("expected $N nodal values, got $(length(V))"))
	return _stack_nodes(Tuple(V))
end
@generated function _stack_nodes(vals::NTuple{N, V}) where {N, V <: StaticArray}
	S = size(V)
	L = prod(S)
	exprs = [:(vals[$n][$s]) for n in 1:N for s in 1:L]
	return :(SArray{Tuple{$(S...), $N}}(tuple($(exprs...))))
end

#--------------------------------------------------------------------------------
# generic element operations
#--------------------------------------------------------------------------------

"""
	get_jacobian(el, ξ, X) -> (J, detJ)

`J = dNdr * X'` (`J[i, j] = ∂xⱼ/∂ξᵢ`) and its determinant. For an element embedded in a
higher-dimensional space (`D < size(X, 1)`) the measure `√det(J Jᵀ)` is returned instead.
"""
function get_jacobian(el::Element{D, N}, ξ::StaticVector{D}, X::StaticMatrix{DX, N}) where {D, N, DX}
	J = shape_function_derivative(el, ξ) * transpose(X)
	return J, _measure(J)
end
get_jacobian(el::Element, ξ, X) = get_jacobian(el, _static_ref(el, ξ), _static_coords(el, X))

@inline _measure(J::SMatrix{D, D}) where {D} = LinearAlgebra.det(J)
@inline _measure(J::SMatrix{D, DX}) where {D, DX} = sqrt(LinearAlgebra.det(J * transpose(J)))

# (DX × D) matrix M with M[j, i] = ∂ξᵢ/∂xⱼ; the pseudo-inverse for embedded elements
@inline _inv_jacobian(J::SMatrix{D, D}) where {D} = LinearAlgebra.inv(J)
@inline _inv_jacobian(J::SMatrix{D, DX}) where {D, DX} = transpose(J) * LinearAlgebra.inv(J * transpose(J))

"""
	interpolate(el, ξ, V, X)

Value of the field with nodal values `V` at the reference point `ξ`.
"""
interpolate(el::Element{D, N}, ξ::StaticVector{D}, V::StaticArray, X) where {D, N} =
	_contract_nodes(V, shape_function(el, ξ))
interpolate(el::Element, ξ, V, X) = interpolate(el, _static_ref(el, ξ), _static_nodal(el, V), X)

function _gradient_with_inv_jacobian(el::Element{D, N}, ξ::StaticVector{D}, V::StaticArray, X::StaticMatrix{DX, N}) where {D, N, DX}
	dNdr = shape_function_derivative(el, ξ)
	invJ = _inv_jacobian(dNdr * transpose(X))
	dNdX = invJ * dNdr
	return _grad_nodes(V, dNdX), invJ
end

"""
	gradient(el, ξ, V, X)

Spatial gradient at `ξ` of the field with nodal values `V` on an element with nodal
coordinates `X`. The spatial axis is appended last: a scalar field gives an `SVector{DX}`,
a vector field an `SMatrix{M,DX}` with `∇u[i, j] = ∂uᵢ/∂xⱼ`.

Line elements return the derivative along the arc length instead (same shape as the value).
"""
gradient(el::Element{D, N}, ξ::StaticVector{D}, V::StaticArray, X::StaticMatrix{DX, N}) where {D, N, DX} =
	first(_gradient_with_inv_jacobian(el, ξ, V, X))
gradient(el::Element, ξ, V, X) = gradient(el, _static_ref(el, ξ), _static_nodal(el, V), _static_coords(el, X))

"""
	hessian(el, ξ, V, X)

Second spatial derivatives at `ξ`; two spatial axes are appended to the value shape
(`H[..., i, j] = ∂²u/∂xᵢ∂xⱼ`). Computed by differentiating the gradient with respect to `ξ`
(ForwardDiff) and mapping with the inverse Jacobian, so it is exact for curved elements.
"""
function hessian(el::Element{D, N}, ξ::StaticVector{D}, V::StaticArray, X::StaticMatrix{DX, N}) where {D, N, DX}
	g(η) = _flatvec(first(_gradient_with_inv_jacobian(el, η, V, X)))
	dg = ForwardDiff.jacobian(g, SVector{D}(ξ))  # (L, D)
	invJ = last(_gradient_with_inv_jacobian(el, ξ, V, X))  # (DX, D)
	H = dg * transpose(invJ)  # (L, DX): Σᵣ dg[l, r] ∂ξᵣ/∂xⱼ
	return _hessian_shape(V, H)
end
hessian(el::Element, ξ, V, X) = hessian(el, _static_ref(el, ξ), _static_nodal(el, V), _static_coords(el, X))

@inline _hessian_shape(::SVector{N}, H::SMatrix{DX, DX}) where {N, DX} = H
@inline _hessian_shape(::SMatrix{M, N}, H::SMatrix{L, DX}) where {M, N, L, DX} = SArray{Tuple{M, DX, DX}}(Tuple(H))
@inline _hessian_shape(::SArray{Tuple{A, B, N}}, H::SMatrix{L, DX}) where {A, B, N, L, DX} = SArray{Tuple{A, B, DX, DX}}(Tuple(H))

"""
	grad(el, ξ, V, X)
	hess(el, ξ, V, X)

Element-level spatial gradient / Hessian — the same as [`gradient`](@ref) / [`hessian`](@ref)
(tatva's `element.gradient`/`element.hessian`), under names that do not clash with the
`gradient`/`hessian` exported by AD packages such as Enzyme or Zygote.
"""
grad(el::Element, ξ, V, X) = gradient(el, ξ, V, X)
hess(el::Element, ξ, V, X) = hessian(el, ξ, V, X)

"""
	get_local_values(el, ξ, V, X) -> (value, gradient, detJ)
"""
function get_local_values(el::Element{D, N}, ξ::StaticVector{D}, V::StaticArray, X::StaticMatrix{DX, N}) where {D, N, DX}
	J, detJ = get_jacobian(el, ξ, X)
	dNdX = _inv_jacobian(J) * shape_function_derivative(el, ξ)
	return _contract_nodes(V, shape_function(el, ξ)), _grad_nodes(V, dNdX), detJ
end
get_local_values(el::Element, ξ, V, X) = get_local_values(el, _static_ref(el, ξ), _static_nodal(el, V), _static_coords(el, X))

#--------------------------------------------------------------------------------
# line elements: arc-length measure and derivative (also when embedded in 2D/3D)
#--------------------------------------------------------------------------------

# (dispatch on `Element{1,N}` so that `N` is tied to the coordinates, which makes these
# methods strictly more specific than the generic ones)

function get_jacobian(el::Element{1, N}, ξ::StaticVector{1}, X::StaticMatrix{DX, N}) where {DX, N}
	tangent = X * transpose(shape_function_derivative(el, ξ))  # SMatrix{DX,1}: dx/dξ
	J = sqrt(sum(abs2, tangent))
	return J, J
end

function gradient(el::Element{1, N}, ξ::StaticVector{1}, V::StaticArray, X::StaticMatrix{DX, N}) where {DX, N}
	J, _ = get_jacobian(el, ξ, X)
	dNdS = SVector{N}(Tuple(shape_function_derivative(el, ξ))) / J
	return _contract_nodes(V, dNdS)
end

function get_local_values(el::Element{1, N}, ξ::StaticVector{1}, V::StaticArray, X::StaticMatrix{DX, N}) where {DX, N}
	J, detJ = get_jacobian(el, ξ, X)
	dNdS = SVector{N}(Tuple(shape_function_derivative(el, ξ))) / J
	return _contract_nodes(V, shape_function(el, ξ)), _contract_nodes(V, dNdS), detJ
end

#--------------------------------------------------------------------------------
# reference-element membership (used to locate points)
#--------------------------------------------------------------------------------

const _SimplexElement = Union{Tri3, Tri6, Tetrahedron4}

"""Whether the reference point `ξ` lies in the reference element (with tolerance `tol`)."""
_is_inside_reference(::_SimplexElement, ξ, tol) = all(x -> x >= -tol, ξ) && sum(ξ) <= 1 + tol
_is_inside_reference(::Element, ξ, tol) = all(x -> abs(x) <= 1 + tol, ξ)

"""Reference-element centroid, a robust starting point for inverse mapping."""
_reference_centroid(el::Element{D, N}) where {D, N} = SVector{D}(sum(reference_nodes(el)[:, n] for n in 1:N) / N)
_reference_centroid(el::_SimplexElement) = (D = reference_dim(el); SVector{D}(ntuple(_ -> 1 / (D + 1), D)))

"""
Points whose convex hull contains the image of the element under its geometry map (the
control points of the map's Bernstein form), or `nothing` if unknown: the nodes of
(multi)linear elements; quadratic elements replace each edge midpoint `m` by the control
point `2m - (a + b)/2` (end nodes `a`, `b`), which bounds the curved edge.
"""
_geometry_control_points(::Union{Line2, Tri3, Quad4, Tetrahedron4, Hexahedron8}, X::StaticMatrix) = X
_geometry_control_points(::Element, ::StaticMatrix) = nothing

_edge_control_point(a, b, m) = 2m - (a + b) / 2

function _geometry_control_points(::Line3, X::StaticMatrix)
	x = ntuple(n -> X[:, n], Val(3))
	return hcat(x[1], x[2], _edge_control_point(x[1], x[2], x[3]))
end

function _geometry_control_points(::Tri6, X::StaticMatrix)
	x = ntuple(n -> X[:, n], Val(6))
	return hcat(x[1], x[2], x[3], _edge_control_point(x[1], x[2], x[4]), _edge_control_point(x[2], x[3], x[5]),
		_edge_control_point(x[3], x[1], x[6]))
end

function _geometry_control_points(::Quad8, X::StaticMatrix)
	x = ntuple(n -> X[:, n], Val(8))
	# the serendipity map is biquadratic; its tensor Bernstein form adds a face control point
	face = (x[5] + x[6] + x[7] + x[8]) - 0.75 * (x[1] + x[2] + x[3] + x[4])
	return hcat(x[1], x[2], x[3], x[4], _edge_control_point(x[1], x[2], x[5]), _edge_control_point(x[2], x[3], x[6]),
		_edge_control_point(x[3], x[4], x[7]), _edge_control_point(x[4], x[1], x[8]), face)
end

"""Number of leading nodes that form the element's (straight-sided) outline."""
_n_corner_nodes(el::Element{D, N}) where {D, N} = N
_n_corner_nodes(::Tri6) = 3
_n_corner_nodes(::Quad8) = 4
_n_corner_nodes(::Line3) = 2

"""`Val(N)` for an element with `N` nodes (a type-stable singleton to capture in closures)."""
@inline _nval(::Element{D, N}) where {D, N} = Val(N)
