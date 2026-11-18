# Reverse-mode gradients with Enzyme, loaded with `using Enzyme`:
#
#     jacrev(energy; backend = AutoEnzyme())(u)                      # one reverse sweep
#     virtual_work_to_residual(G; test_size = n, backend = AutoEnzyme())
#
# Second derivatives are not taken through Enzyme: with the Enzyme versions this was written
# against, nested differentiation of the operator code (forward-over-reverse, or reverse
# through ForwardDiff duals) crashes Enzyme's compiler. Sparse Hessians/tangents use
# ForwardDiff — `jacfwd`/`sparse_hessian` for general energies, `ElementAssembler` for
# element-wise energies at scale.

module TatvaEnzymeExt

using Tatva
using Tatva: ArgDerivative, VirtualWorkResidual, ColoredJacobian
using ADTypes: AutoEnzyme
using Enzyme

# Runtime activity is on by default: the element loops capture constant mesh data next to
# active field data in the same closures, which static activity analysis cannot separate.
_mode(::AutoEnzyme{Nothing}) = Enzyme.set_runtime_activity(Enzyme.Reverse)
_mode(b::AutoEnzyme) = b.mode

function Tatva.derivative(backend::AutoEnzyme, f::F, x::AbstractArray{<:AbstractFloat}) where {F}
	# scalar output: one reverse sweep (call f first only if its return type is not inferred)
	R = Core.Compiler.return_type(f, Tuple{typeof(x)})
	if (R <: AbstractFloat) || (!isconcretetype(R) && f(x) isa AbstractFloat)
		dx = Enzyme.make_zero(x)
		Enzyme.autodiff(_mode(backend), Const(f), Active, Duplicated(x, dx))
		return dx
	end
	# array-valued: reverse-mode Jacobian (one sweep per output)
	y = f(x)
	J = only(Enzyme.jacobian(_mode(backend), f, x))
	return reshape(J, size(y)..., size(x)...)
end

Tatva.derivative(backend::AutoEnzyme, f::F, x::AbstractFloat) where {F} =
	only(only(Enzyme.autodiff(_mode(backend), Const(f), Active, Active(x))))

const _NO_SECOND_ORDER = "sparse second derivatives through the Enzyme backend are not supported " *
						 "(nested Enzyme differentiation of the operator code is unreliable); use the default " *
						 "AutoForwardDiff() backend for `jacfwd`/`sparse_hessian`, or `ElementAssembler` for " *
						 "element-wise energies"

(cj::ColoredJacobian{<:ArgDerivative{F, I, <:AutoEnzyme}})(u::AbstractVector, args...; kwargs...) where {F, I} =
	throw(ArgumentError(_NO_SECOND_ORDER))
(cj::ColoredJacobian{<:VirtualWorkResidual{F, T, <:AutoEnzyme}})(u::AbstractVector, args...; kwargs...) where {F, T} =
	throw(ArgumentError(_NO_SECOND_ORDER))

end # module TatvaEnzymeExt
