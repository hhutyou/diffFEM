# Tatva.jl

A Julia port of [tatva](https://github.com/smec-ethz/tatva) (तत्त्व) — Lego-like building
blocks for **differentiable, energy-based finite element analysis**. Write the energy (or the
virtual work) of your problem; automatic differentiation provides residuals and sparse tangent
matrices.

The port keeps tatva's building blocks and API structure:

| tatva (Python/JAX) | Tatva.jl |
| --- | --- |
| `tatva.element`: `Line2, Line3, Tri3, Tri6, Quad4, Quad8, Tetrahedron4, Hexahedron8` | same names; `shape_function`, `gradient`/`grad`, `hessian`/`hess`, `get_jacobian`, … |
| `Mesh`, `Mesh.unit_square`, `Mesh.rectangle`, `hmin/hmax`, `extract_local_mesh`, `find_containing_polygons` | `Mesh`, `unit_square`, `rectangle`, … |
| `Operator.eval / grad / hess / integrate / integrate_per_element / map / map_over_elements / interpolate / make_interpolate / project / quads` | `evaluate`, `grad`, `hess`, `integrate`, `integrate_per_element`, `map_over_quads`, `map_over_elements`, `interpolate`, `make_interpolate`, `project`, `quads` |
| `tatva.linalg`: `inv`, `det`, `contract`, `einsum` | `Tatva.Linalg` |
| `sparse.ColoredMatrix`, `sparse.jacfwd`, `sparse.linearized_jacfwd`, distance-2 coloring | `ColoredMatrix`, `jacfwd`, `linearized_jacfwd`, `sparse_hessian`, `distance2_colors`, … |
| `sparse.pattern_from_mesh / _compound / _energy / _virtual_work` | same names |
| `lifter`: `Lifter`, `Fixed`, `Periodic`, `PeriodicMPI`, `RuntimeValue`, `lifted`, `reduce_adjoint`, `adapt_sparsity` | same names (`reduce(lifter, u)` for `lifter.reduce(u)`) |
| `compound`: `Compound`, `field`, `FieldSize.AUTO`, `stack_fields`, `Nodal/Local/Shared` | `Compound(...)` / `@compound`, `field`, `AUTO`, `stack_fields`, … |
| `mpi`: `ExchangePlan`, `AllreducePlan` (mpi4py/mpi4jax) | same, via MPI.jl (package extension) |
| `utils`: `virtual_work_to_residual`, `make_project_function`, `create_g2l` | same names |

plus Julia additions: `ElementAssembler` (element-wise, optionally threaded energies,
gradients and sparse Hessians, see *Scaling up*), `EnergyProblem` and `newton` (nonlinear
solves, see *Solving nonlinear problems*), and mesh input from Gmsh and output for ParaView.

## Quick start

```julia
using Tatva, LinearAlgebra

mesh = unit_square(20, 20)                 # Tri3 mesh, coords (2, n_nodes)
op = Operator(mesh, Tri3())

const μ, λ = 1.0, 1.5
ψ(∇u) = (ε = (∇u + ∇u') / 2; μ * dot(ε, ε) + λ / 2 * tr(ε)^2)   # ∇u is an SMatrix{2,2}
energy(u) = integrate(op, ψ.(grad(op, reshape(u, 2, :))))

u = zeros(2 * num_nodes(mesh))
r = jacrev(energy)(u)                                  # residual ∂E/∂u
cm = ColoredMatrix(pattern_from_mesh(mesh, 2))         # or pattern_from_energy(energy, length(u))
K = sparse_hessian(energy, cm)(u)                      # ∂²E/∂u², colored sparse AD
Δu = spsolve(K, -r)                                    # sparse(K) is a SparseMatrixCSC
```

Boundary conditions with a `Lifter`, several fields with a `Compound`:

```julia
Solution = Compound(:Solution, :u => field((2, AUTO)); mesh = mesh)  # AUTO = number of nodes
left, right = findall(mesh.coords[1, :] .== 0), findall(mesh.coords[1, :] .== 1)
lifter = Lifter(Solution.size, Fixed(Solution.u[:, left], 0.0),
                Fixed(Solution.u[1, right], RuntimeValue(:stretch, 0.0)))
energy_free(u_free, lf) = integrate(op, ψ.(grad(op, Solution(lift_from_zeros(lf, u_free)).u)))
cm = ColoredMatrix(adapt_sparsity(lifter, pattern_from_mesh(mesh, 2)))
K = sparse_hessian(energy_free, cm)(u_free, with_values(lifter, :stretch => 0.1))
```

See `examples/` and `test/test_examples.jl` (patch test, quadratic Newton convergence, the
derivative of a compliance *through* the sparse solve).

## Conventions (the one thing to know when coming from tatva)

tatva's arrays are row-major; Julia's are column-major. The port **reverses every axis**, so
the flat degree-of-freedom numbering is *identical* to tatva's:

- `mesh.coords` is `(dim, n_nodes)`, `mesh.elements` is `(nodes_per_element, n_elements)`;
- nodal fields have the node axis last: `(n_nodes,)`, `(n_comp, n_nodes)`, `(a, b, n_nodes)` —
  e.g. tatva's `u_flat.reshape(-1, 2)` is `reshape(u_flat, 2, :)`;
- compound field shapes are reversed: tatva's `field((AUTO, 3))` is `field((3, AUTO))`, and
  `Solution.u[nodes, :]` is `Solution.u[:, nodes]`;
- fields at quadrature points are `Matrix{V}` of size `(n_quad, n_elements)` whose entries are
  numbers or `StaticArray`s **in natural orientation** (`∇u[i, j] = ∂uᵢ/∂xⱼ`), so energy
  densities are plain point-wise functions broadcast over the matrix (`ψ.(grad(op, u))`) —
  this replaces tatva's `autovmap`;
- every index is 1-based (MPI ranks stay 0-based).

## Automatic differentiation

Every function in the package is generic in its number type, so energies can be
differentiated with respect to the unknowns, the mesh coordinates, Dirichlet values or
material parameters, also through `spsolve`.

- `jacrev(f)` / `jacfwd(f, cm)` / `linearized_jacfwd` / `sparse_hessian` reproduce tatva's
  global API with **ForwardDiff**: `jacfwd` seeds one dual partial per color of a
  `ColoredMatrix` (`color_batch_size` colors per evaluation); `jacrev` evaluates the gradient
  in chunks of forward passes, so its cost grows like `n_dofs²`. Nesting (sparse Jacobian of a
  gradient, tangents of virtual work, derivatives through solves) is handled by tagging dual
  numbers with the argument types.
- `sparse_hessian` colors the symmetric Hessian with an acyclic coloring of
  [SparseMatrixColorings.jl](https://github.com/gdalle/SparseMatrixColorings.jl): 8 colors instead
  of the 18 of the distance-2 coloring for a 2D elasticity mesh, 30 instead of 87 for 3D
  tetrahedra, and one gradient evaluation per color (`coloring = :column` is tatva's way).
- With `using Enzyme`, `jacrev(f; backend = AutoEnzyme())` computes gradients in one **reverse**
  sweep. Second derivatives are not taken through Enzyme: nested Enzyme differentiation of the
  operator code crashed Enzyme's compiler (0.13 on Julia 1.12), so asking for one raises an
  error instead.

### Scaling up: element-wise assembly

For energies of the form `E(u) = Σₑ Σ_q w_q detJ_q density(ξ_q, Uₑ, extrasₑ...)`,
`ElementAssembler` differentiates each element with ForwardDiff (nested duals give the
element energy, gradient and Hessian in one pass) and scatters into the mesh pattern:

```julia
asm = ElementAssembler(op, 2)                          # 2 dofs per node
density(ξ, Ue, Xe) = ψ(grad(op.element, ξ, Ue, Xe))    # element-level grad
E, g, K = assemble(asm, density, reshape(u, 2, :), mesh.coords)
r, Kr = reduce_adjoint(lifter, g), reduce_hessian(lifter, K)   # free dofs
E = assemble!(g, K, asm, density, reshape(u, 2, :), mesh.coords)  # next Newton step: refill g, K
```

`ElementAssembler(op, 2; threaded = true)` assembles in parallel tasks when Julia runs with
several threads (`julia -t auto`): blocks of consecutive elements are colored so that blocks
assembled at the same time share no node (the density must then be safe to call
concurrently). With `using HyperHessians`, element Hessians use the hyper-dual numbers of
[HyperHessians.jl](https://github.com/KristofferC/HyperHessians.jl), 1.1–1.3× faster than
nested dual numbers (nested duals remain where an enclosing ForwardDiff derivative passes
through). It is optional because HyperHessians 0.3 still holds LogExpFunctions at 0.3.

Linear elasticity on `unit_square(n, n)` (Tri3, 8 Hessian colors), median times on an Apple
M-series laptop (`julia -t 8 benchmark/colored_matrix_benchmark.jl`; "all" = energy, gradient
and Hessian):

| mesh | dofs | global ForwardDiff: gradient / Hessian | Enzyme gradient | element-wise: gradient / all | all, 8 threads |
| --- | --- | --- | --- | --- | --- |
| 20×20 | 882 | 13 ms / 137 ms | 1.5 ms | 0.05 ms / 0.3 ms | 0.3 ms |
| 100×100 | 20 402 | — | 38 ms | 1.2 ms / 7.5 ms | 1.6 ms |
| 200×200 | 80 802 | — | 153 ms | 4.9 ms / 30 ms | 5.8 ms |

So: the global API for prototyping and for energies that do not split into elements; the
assembler (and Enzyme gradients) for production-size meshes.

### Compared with tatva (JAX)

`benchmark/compare_with_tatva/` runs the same problem in both (linear elasticity on
`unit_square(n, n)`, the same displacement state; energy, gradient norm and Hessian norm
agree to 12 digits): tatva 0.11.6 on JAX 0.12 (`jax.jit`, float64, CPU) and Tatva.jl on
Julia 1.12, Apple M-series laptop, median run times:

| dofs | energy: tatva / Julia | gradient: tatva / Julia, 1 thread / 8 threads | Hessian: tatva / Julia (energy + gradient + Hessian), 1 thread / 8 threads |
| --- | --- | --- | --- |
| 20 402 | 0.55 / 0.76 ms | 0.93 / 1.2 / 0.69 ms | 7.5 / 7.6 / 1.9 ms |
| 80 802 | 1.2 / 3.0 ms | 2.0 / 5.0 / 0.95 ms | 44 / 31 / 5.7 ms |
| 157 922 | 1.5 / 5.9 ms | 3.6 / 9.6 / 1.7 ms | 79 / 61 / 12 ms |

XLA ran these on 2.5–5 cores. JAX compiles again for every mesh size (0.1–0.8 s per
function and size); Julia compiles once per session. tatva's global formulation is the fast
one in JAX: in Julia it is not (ForwardDiff gradients cost `O(n²)`, the Enzyme gradient of
the global energy is about 90× slower than `jax.grad`), hence the element-wise assembler.

Material parameters read by a density from global scope must be `const` (or passed in as
arguments): with plain globals every quadrature-point evaluation is dynamically dispatched,
which made the element-wise columns above 5–10× slower.

## Solving nonlinear problems

`EnergyProblem` makes the element-wise energy a function of the free dofs of a lifter: its
gradient and Hessian are assembled straight into the free dofs (`Pᵀ g`, `Pᵀ K P`, no full
matrix), and `newton` minimizes it:

```julia
problem = EnergyProblem(asm, density, lifter, mesh.coords)
solver = CholeskySolver()  # keep it: one symbolic factorization for all solves
u = zeros(lifter.size_reduced)
for stretch ∈ 0.2:0.2:1.0  # load steps
    result = newton(with_values(problem, :stretch => stretch), u; linear_solver = solver)
    u = result.u
end
```

- Linear solvers keep their factorization: while the sparsity pattern stays the same, only
  the numerical factorization is redone. `CholeskySolver()` (CHOLMOD; the default, LU where
  the Hessian is not positive definite), `DirectSolver()` (UMFPACK LU), and with
  `using LinearSolve` any [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl) algorithm,
  e.g. `LinearSolveSolver(KLUFactorization())`, or conjugate gradients with algebraic multigrid
  ([AlgebraicMultigrid.jl](https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl)) for
  large 3D problems:
  `LinearSolveSolver(KrylovJL_CG(); preconditioner = K -> aspreconditioner(smoothed_aggregation(K; B)))`
  with `B = rigid_body_modes(mesh.coords; lifter)`.
- The line search makes Newton's method a minimizer: where the Newton step does not decrease
  the energy (an indefinite Hessian), the step of `K + τI` is taken (modified Newton), and
  steps are shortened until the energy decreases enough — also where elements would turn
  inside out.

The 3D bar of `examples/hyperelasticity.jl`, 5 load steps to 25 % elongation, total solve time
(`benchmark/newton_benchmark.jl`):

| free dofs | assemble, reduce, fresh LU (no line search) | `newton`, LU | `newton`, Cholesky | `newton`, Cholesky, 8 threads |
| --- | --- | --- | --- | --- |
| 7 533 | 4.9 s | 4.5 s | 3.2 s | 2.8 s |
| 23 829 | fails (inverted elements) | 24 s | 14 s | 10 s |

The factorizations dominate. For one factorization of 3D linear elasticity (Tet4, 70 644 free
dofs): LU 4.9 s, Cholesky 1.4 s, CG with multigrid and rigid body modes 1.5 s (24
iterations, 114 without the modes) — multigrid scales linearly and wins on larger meshes.

## Mesh input and output

- `using Gmsh` loads `read_gmsh("part.msh")` (also `.geo` and other geometries, which are
  meshed, or `read_gmsh()` for the current model of a Gmsh session): the `Mesh`, the element
  (`Tri3()`, `Tetrahedron4()`, …) and the physical groups as node and element sets for
  boundary conditions and materials, e.g. `groups["left"].nodes`.
- `using WriteVTK` loads `write_vtk("result", op; point_data = (u = U,), cell_data = (…))` for
  ParaView; `vtk_grid("step", op)` gives WriteVTK's file object, e.g. for time series with
  `paraview_collection`.

## Sparsity detection

`pattern_from_energy` / `pattern_from_virtual_work` trace the energy with
[SparseConnectivityTracer.jl](https://github.com/adrhill/SparseConnectivityTracer.jl)
(operator overloading instead of tatva's jaxpr interpreter). The result is a pattern valid for
every input (no false negatives). Loops, gathers/scatters, closures and custom-derivative
functions are traced as ordinary code — including data-dependent loop carries, which tatva's
`scan` handler cannot track. Two helpers cover what a tracer cannot run:

- `branch(pred, f, g, args...)` / `switch(i, branches, args...)` instead of `if` on traced
  values (both branches are traced and unioned, like tatva's `cond` handler; `ifelse` also works);
- `opaque(f, args...)` for code that cannot run on tracers (foreign libraries, `eigvals`,
  `sort`): all inputs of the call are coupled (tatva's rule for callbacks and FFI calls). Call it
  per quadrature point to keep the pattern sparse (tatva's `register_elementwise_ffi`).

## MPI

`using MPI` loads the extension: `Compound(...; partition_info, comm)` builds the distributed
dof layout, `adapt_layout(lifter, layout, comm)` resolves constraints (incl. `PeriodicMPI`
across ranks), `ExchangePlan` gathers ghosts and assembles owned rows of gradients and sparse
Hessians point-to-point, `AllreducePlan` does the same with replicated vectors.
`test/mpi/test_parallel_assembly.jl` checks a partitioned assembly entry by entry against the
serial one.

## Deliberate differences from tatva

Fixes and generalisations:

- coloring uses the column-intersection graph `AᵀA` (equal to tatva's `A²` for symmetric
  patterns, correct also for the unsymmetric tangents of virtual-work forms);
- `project` with a lifter accounts for non-zero fixed values (tatva drops `M_fc g` from the
  right-hand side, which is only correct for homogeneous constraints) and takes its test
  functions from the constrained space (tatva leaves periodic slaves at zero in them);
- periodic constraints: `reduce_adjoint` is the exact transpose of the lift also when a
  master is a slave of the same constraint (tatva adds to the masters before clearing the
  slaves), and `adapt_sparsity` pulls the pattern back through the constraints in reverse
  order, so chains (a slave that is the master of a later constraint) keep all couplings;
  a dof listed twice in one `Periodic` is an error;
- `hmin`/`hmax` do not take a quadrilateral in 3D for a tetrahedron (or a quadratic line for
  a triangle), and accept the element: `hmax(mesh, element)`, `hmax(op)`;
- `make_interpolate` maps points with Newton's method (exact for bilinear and curved elements,
  works in 3D; tatva does one linear step and 2D ray casting only); candidate elements come
  from a grid of bounding boxes that also contain curved elements (Bernstein control points);
- surface elements (`Tri3` in 3D, …) integrate with the Gram determinant `√det(J Jᵀ)`;
- `add(lifter, constraint)` keeps the runtime values already set;
- `sparse_hessian` uses a symmetric coloring (fewer colors than tatva's distance-2 one).

Not ported, because they only exist for JAX/XLA: `jit` and static arguments, pytree
registration, `batch_size` (there is no vmap memory blow-up to bound), the XLA/HLO
regression tests of `linalg` (replaced by allocation and type-stability tests of the element
kernels), `to_bcoo`/`to_bcsr`, the tracer's internal data structures and `_unwrap_jit`.

## Tests

```sh
julia --project -e 'using Pkg; Pkg.test()'
```

Every tatva test module has a Julia counterpart in `test/` (each file states what was ported and
what has no Julia meaning); the Enzyme tests run in a separate process, the MPI tests on 2 ranks
with `mpiexec`, and `test_extensions.jl` covers Gmsh, WriteVTK and HyperHessians. Benchmarks
(`benchmark/`) mirror tatva's `*_benchmark.py`, plus the comparison with tatva and the Newton
solves.

## License

GNU Lesser General Public License v3.0 or later (see `COPYING` and `COPYING.LESSER`), as the
original: tatva © 2025 ETH Zurich (SMEC), Mohit Pundir and Flavio Lorez. This is an
independent port.
