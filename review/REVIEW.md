Review of Tatva.jl — 2026-10-22

Reviewed all source modules and both extensions, with supporting inspection of tests, examples, and benchmarks. Package source and existing tests were not changed. Reproductions and measurement code are in [probes.jl](probes.jl).

Validation: `julia --project=. --startup-file=no test/runtests.jl` passed **880/880 tests** on Julia 1.12.6 in 3m05.8s. MPI and Enzyme tests were skipped by the runner because those packages are unavailable in the active environment. The findings below expose cases missing from the passing suite; passing it does not establish correctness for those cases.

Run from the Tatva.jl directory:

```sh
julia --project=. --startup-file=no review/probes.jl --correctness-only
julia --project=. --startup-file=no review/probes.jl
```

The probe prints observed and expected results; it is a diagnostic script, not a passing regression suite. It does not modify package source. Its sparse-matrix mutation example only mutates objects constructed within that probe.

**Correctness findings, in suggested repair order**

1. **High: chained periodic constraints lose required Hessian entries.** Location: [src/lifter.jl](../src/lifter.jl), lines 331–339. Reproduced with `Lifter(4, Periodic(2,1), Periodic(3,2))` and a full pattern containing only `(3,3)`. `adapt_sparsity` returns an entirely empty reduced pattern, although `P' * S * P`, with `P = lifting_matrix(lifter)`, contains `(1,1)`. Lifting composes constraints forwards, so pulling back a Hessian must apply their transformations in reverse order. The current loop applies them forwards. This can silently omit tangent entries used by colored differentiation. Reverse the constraint traversal and test against the structural pattern of `P' * S * P` for chained constraints.

2. **High: periodic L2 projection uses the wrong test space.** Location: [src/utils.jl](../src/utils.jl), lines 104–108. `_full_test` fills only free dofs and leaves periodic slaves at zero, whereas `_full_trial` applies periodic constraints. Consequently the solve does not form the required `P' * M * P` and `P' * b`. On a 2×2 Quad4 mesh, projecting `x²+y` with the right boundary tied to the left gives a maximum nodal error of **0.6666667** against a directly assembled constrained mass solve; the reduced residual norm is **0.102062**, rather than numerical zero. Lift test functions with the homogeneous linear map `P`; do not include nonzero Dirichlet offsets in the test functions.

3. **High: mutating an assembled Hessian corrupts the reusable assembler.** Location: [src/assembly.jl](../src/assembly.jl), line 187. Returned matrices share `colptr` and `rowval` with `asm.pattern`. For an energy with a diagonal Hessian, calling `dropzeros!(K)` shrank the assembler's `rowval` from 14 to 4 while its `nzval` still had length 14. Previously returned matrices also share the structure, and `asm.positions` becomes stale. Later assembly/solves can therefore be wrong or fail. Copy structural arrays for independently mutable return values, or expose a documented workspace API that owns its output and explicitly restricts structural mutation.

4. **High: overlapping periodic assignments have an incorrect adjoint.** Location: [src/lifter.jl](../src/lifter.jl), lines 137–142. The transpose operation adds slave contributions to masters and then zeroes slaves. If a master is also a slave in the same constraint, the added contribution is erased. With `Lifter(4, Periodic(2,1), Periodic([2,3],[1,2]))` and residual `[1,2,3,4]`, `reduce_adjoint` returns `[3,4]`, but `lifting_matrix(l)' * r` is `[6,4]`. Gather the slave residuals, clear slave entries, and then scatter-add to masters. Also define or reject duplicate slave assignments, whose transpose needs to respect overwrite semantics.

5. **High, established by code inspection; not executed: MPI layout adaptation has inconsistent collective participation.** Location: [ext/TatvaMPIExt.jl](../ext/TatvaMPIExt.jl), lines 198–202. `_create_dof_layout` performs `Allgather` and `Allreduce!`, but `adapt_layout` calls it only on ranks with nonempty `local_extra`. A periodic master missing on one rank but already present on another sends those ranks through different collective sequences, risking a hang or collective mismatch. Make the decision collectively, then have every rank participate in rebuilding the layout, including ranks adding zero ghosts. The existing periodic MPI test has every relevant master already locally present and does not cover this branch. Validate with asymmetric ghost requirements on two ranks after fixing it.

6. **Medium: curved-element point location rejects interior points.** Location: [src/operator.jl](../src/operator.jl), lines 389–399. Bounding boxes use only nodal coordinate extrema. Quadratic shape functions can produce coordinates outside those extrema; the point is rejected before Newton mapping. The Tri6 probe has an interior reference point `(0.2,0.001)` mapping to approximately `(0.2,-0.0388)`, with positive Jacobian determinant 1.2, but all nodal y coordinates are nonnegative. `make_interpolate` incorrectly throws “outside the mesh.” Use conservative polynomial bounds, or bypass this rejection for curved elements until such bounds are available. A larger fixed padding does not generally solve the problem.

7. **Medium: embedded quadrilateral meshes get incorrect element diameters.** Location: [src/mesh.jl](../src/mesh.jl), lines 125–138. The routine infers element topology from node count and ambient dimension, treating any four-node element in 3D as a tetrahedron. `hmax(unit_square(1,1; type=:quad, dim=3))` returns **0.0**, rather than `sqrt(2)`. This can invalidate mesh-dependent tolerances or time steps. Accept element/topology information rather than inferring it solely from these two sizes. The analogous three-node heuristic also confuses quadratic line elements with triangles.

8. **Medium, code inspection: unchecked gathers accept incorrectly sized auxiliary fields.** Locations: [src/operator.jl](../src/operator.jl), lines 97–113 and 182–207; [src/assembly.jl](../src/assembly.jl), lines 99–106. `grad`/`evaluate` validate their nodal input, but public map methods do not validate all gatherers, and assembly validates only `u`, not `extras`. The generated gatherers then index with `@inbounds`. A short nodal coefficient or per-element array can cause invalid reads rather than a useful dimension error. Validate every nodal/per-element array once at the public call boundary, preserving unchecked indexing inside validated loops. This finding was not exercised with unchecked out-of-bounds reads.

**Measured allocation and runtime findings**

Measurements use a simple quadratic vector-field gradient energy on Tri3 meshes, two warm-up calls, median of seven calls (three for the global Hessian), and `@allocated` for bytes per call. These are local microbenchmarks, not comprehensive application timings; compilation is excluded and allocation totals are not peak resident memory. Times vary: the global Hessian measured 201–292 ms across two runs.

| Mesh / operation | Median time, final run | Allocation per call |
| --- | ---: | ---: |
| 20×20, global `jacrev` gradient | 10.127 ms | 55.01 MB |
| 20×20, global `sparse_hessian` | 292.287 ms | 948.93 MB |
| 20×20, `assemble_gradient` | 0.033 ms | 8.56 KB |
| 20×20, `assemble` (energy, gradient, Hessian) | 0.184 ms | 108.00 KB |
| 100×100, `assemble_gradient` | 0.777 ms | 165.23 KB |
| 100×100, `assemble` | 4.691 ms | 2.43 MB |
| 100×100, `assemble_energy` | 1.320 ms | 1.28 MB |
| 100×100, `reduce_hessian` | 3.538 ms | 21.16 MB |
| 100×100, reduction with precomputed `P` | 3.666 ms | 16.16 MB |
| 100×100, Dirichlet-only `K[free,free]` | 1.871 ms | 4.85 MB |

MB/KB above are decimal units. The reduction benchmark uses only fixed constraints, and the probe verifies that direct slicing agrees with `reduce_hessian`. Direct slicing is not valid for general periodic constraints. Caching `P` reduced allocation but did not show a reliable speed improvement in these short measurements.

**Recommended performance work**

1. **Remove the dense sparse-solve fallback for common numeric types.** [src/sparse/colored_matrix.jl](../src/sparse/colored_matrix.jl), lines 258–259, calls `Matrix(A) \ b` for every element type except Float64 and ComplexF64. Thus Float32 and integer sparse matrices become dense. A 1000×1000 diagonal Float32 solve allocated **4,038,864 bytes** and took 0.582 ms, versus **521,344 bytes** and 0.046 ms for Float64. At 100,000 unknowns, the Float32 dense array alone requires 40 GB. Convert supported lower-precision/integer matrices to an appropriate sparse solver type, or use a compatible sparse backend; make unsupported precision behavior explicit. This is the most urgent allocation-related implementation issue.

2. **Use element assembly for element-separable energies.** The existing implementation is already effective: gradient allocations are close to output-vector size, and assembling all three quantities is vastly cheaper than the global forward-over-forward Hessian. This is an algorithmic distinction, not something broadcasting micro-optimizations can eliminate. Preserve the global API for genuinely nonlocal energies. Add reusable `assemble!`/`assemble_gradient!` workspaces for repeated Newton steps, with buffers specialized to the number type so outer AD remains supported.

3. **Precompute constraint reduction and specialize fixed constraints.** [src/assembly.jl](../src/assembly.jl), lines 206–247, rebuilds `P` with a separate small vector per full dof on every `reduce_hessian` call. Cache it while the constraint mapping is unchanged. For Dirichlet-only constraints, use a precomputed sparse restriction/scatter or direct slicing. For periodic constraints, precompute reduced assembly positions so element contributions can accumulate directly into the reduced pattern. The current reduction allocated considerably more than assembly itself.

4. **Reuse the primal sparse factorization when differentiating a solve.** [src/sparse/colored_matrix.jl](../src/sparse/colored_matrix.jl), lines 245–252, solves once for the primal and again for batched tangent RHSs using the same `Av`. In the ordinary Float64 primal case, each call executes `lu(Av)`. Factor once and reuse that factorization for both solves. Preserve the recursive behavior for nested dual numbers. Tangent sparse values, RHS matrices, and sliced partial arrays are further workspace candidates.

5. **Stream colored derivatives into nonzero storage.** [src/sparse/colored_matrix.jl](../src/sparse/colored_matrix.jl), lines 169–203, always materializes a dense `n_outputs × n_colors` compressed Jacobian, constructs a new nonzero vector, and copies that vector again through `with_data`. Color batching does not bound the full compressed-matrix allocation. Extract each batch directly into the final nonzero buffer, and compute the primal output only for `linearized_jacfwd`. A cached CSR-to-CSC value permutation would also avoid rebuilding/transposing sparse structure on every `spsolve(cm,b)`.

6. **Fuse pure energy accumulation and cache fixed geometry.** `assemble_energy` currently constructs the full quadrature-value matrix and a per-element integral vector before summing. Accumulating each element's scalar energy directly avoids these temporaries. For fixed meshes, cache quadrature weights and physical shape gradients; invalidate/recompute them when coordinates change, and preserve a geometry-differentiable path. The existing static element kernels and precomputed assembly positions are worth retaining.

Other possible work is lower priority without profiling a representative application: replace repeated string parsing and `Dict{String,Function}` dispatch in `Linalg.contract` if it is called inside user densities; use a spatial index for repeated point location; and consider parallel element assembly only after providing race-free accumulation. These are source-based opportunities, not measured speedup claims.

Suggested order: fix constraint correctness and sparse-structure ownership first; address MPI collective participation before distributed use; eliminate dense sparse solves; then optimize constraint reduction, factorization reuse, and reusable assembly buffers. Add regression coverage for the reproduced cases before tuning their implementations.
