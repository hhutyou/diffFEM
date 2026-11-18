# Element Hessians with hyper-dual numbers (HyperHessians.jl), loaded with `using HyperHessians`:
# value, gradient and Hessian of an element energy in one pass, about 1.1–1.3× faster than
# the nested ForwardDiff duals of the core. Used when the element energy computes in plain
# floating point; with dual numbers of an enclosing ForwardDiff derivative in play (shape or
# parameter sensitivities), the nested duals keep their tags apart.

module TatvaHyperHessiansExt

using Tatva: Tatva, _ElementEnergy
using HyperHessians
using StaticArrays: SVector

@inline function Tatva._value_gradient_hessian(f::_ElementEnergy, x::SVector{L, T}) where {L, T <: AbstractFloat}
	if Base.promote_op(f, typeof(x)) <: AbstractFloat
		r = HyperHessians.hessian_gradient_value(f, x)
		return r.value, r.gradient, r.hessian
	end
	return Tatva._value_gradient_hessian_nested(f, x)
end

end # module
