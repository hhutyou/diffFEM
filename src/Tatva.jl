# Copyright (C) 2025 ETH Zurich (SMEC) — original Python/JAX implementation (tatva)
# Julia port distributed under the same terms.
#
# This file is part of Tatva.jl, a Julia port of tatva.
#
# Tatva.jl is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Tatva.jl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.

"""
	Tatva

Lego-like building blocks for differentiable finite element analysis — a Julia port of
the Python/JAX library `tatva` (तत्त्व).

Conventions (the Julia translation of tatva's row-major arrays reverses every axis, so the
flat degree-of-freedom numbering is identical to tatva's):

- mesh coordinates are `(dim, n_nodes)`, connectivity is `(nodes_per_element, n_elements)`,
  all indices are 1-based;
- nodal fields are arrays with the node axis **last**: `(n_nodes,)` for scalars,
  `(n_comp, n_nodes)` for vectors, `(a, b, n_nodes)` for tensors (or a vector of
  `StaticArray`s);
- fields at quadrature points are `Matrix{V}` of size `(n_quad, n_elements)` whose entries
  `V` are numbers or `StaticArray`s in natural orientation, e.g. a displacement gradient is
  an `SMatrix` with `∇u[i, j] = ∂uᵢ/∂xⱼ`.
"""
module Tatva

using LinearAlgebra
using SparseArrays
using StaticArrays
using ForwardDiff
import SparseMatrixColorings as SMC
using UUIDs: UUID, uuid4
using ADTypes: ADTypes, AbstractADType, AutoForwardDiff, AutoEnzyme

include("linalg.jl")
include("elements.jl")
include("mesh.jl")
include("operator.jl")
include("ad.jl")
include("sparse/coloring.jl")
include("sparse/colored_matrix.jl")
include("lifter.jl")
include("compound.jl")
include("sparse/patterns.jl")
include("sparse/tracer.jl")
include("utils.jl")
include("assembly.jl")
include("solvers.jl")
include("io.jl")
include("mpi.jl")

# elements
export Element, Line2, Line3, Tri3, Tri6, Quad4, Quad8, Tetrahedron4, Hexahedron8
export reference_nodes, shape_function, shape_function_derivative, get_jacobian
export interpolate, gradient, hessian, get_local_values
export reference_dim, nodes_per_element, num_quad_points

# mesh
export Mesh, PartitionInfo, unit_square, rectangle, num_nodes, num_elements, spatial_dim
export set_coords, hmin, hmax, extract_local_mesh, find_containing_polygons

# operator
export Operator, map_over_quads, map_over_elements, integrate, integrate_per_element
export evaluate, grad, hess, quads, make_interpolate, project, get_integration_weights

# automatic differentiation & sparse differentiation
export jacrev, jacfwd, linearized_jacfwd, sparse_hessian, AutoForwardDiff, AutoEnzyme
export ColoredMatrix, n_colors, to_dense, with_data, spsolve
export distance2_colors, distance2_color_and_seeds, seeds_from_coloring
export largest_degree_first_distance2_colors, smallest_last_distance2_colors
export pattern_from_mesh, pattern_from_compound, pattern_from_energy, pattern_from_virtual_work
export branch, switch, opaque

# lifter
export Lifter, Constraint, Fixed, Periodic, PeriodicMPI, RuntimeValue, LifterError
export lift, lift_from_zeros, reduce_adjoint, with_values, lifted
export augment_sparsity, reduce_sparsity, adapt_sparsity

# compound
export Compound, CompoundLayout, CompoundState, CompoundError, field, AUTO, stack_fields
export Nodal, Local, Shared, LOCAL, NODAL, SHARED, set, indices, flatten, @compound, extend

# utils
export virtual_work_to_residual, make_project_function, create_g2l

# element-wise assembly (scalable second derivatives)
export ElementAssembler, assemble, assemble!, assemble_energy, assemble_gradient, assemble_gradient!, assemble_hessian
export lifting_matrix, reduce_hessian

# solvers
export EnergyProblem, newton, NewtonResult, LinearSolver, DirectSolver, CholeskySolver, LinearSolveSolver
export rigid_body_modes

# mesh input, result output (implemented in the Gmsh and WriteVTK package extensions)
export read_gmsh, write_vtk

# mpi (implemented in the MPI package extension)
export ExchangePlan, AllreducePlan, make_scatter_fwd_set, make_scatter_rev_add
export make_allgather, make_allreduce_owned, adapt_layout, get_layout, global_indices, global_view

end # module Tatva
