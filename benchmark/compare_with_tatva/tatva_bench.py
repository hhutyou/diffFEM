"""Head-to-head with Tatva.jl (see julia_bench.jl in this folder): the same 2D linear-elastic
energy on Mesh.unit_square(n, n) (Tri3), the same displacement state, timed the way tatva's
own benchmarks do (jax.jit, float64, median of 5 runs after the compiling first call).

    python tatva_bench.py            # in an environment with tatva (pip install -e <tatva>)

Prints, per mesh: energy / gradient (jax.grad) / sparse Hessian (tatva.sparse.jacfwd of the
gradient, all colors in one batch) - compile time (first call) and run time - and check values
(E, |g|, |K|_F) that julia_bench.jl must reproduce. "cpu/wall" is the process CPU time over
the wall time of the timed runs (> 1: XLA used several threads).
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
from jax_autovmap import autovmap

from tatva import Mesh, Operator, element
from tatva.sparse import ColoredMatrix, jacfwd, pattern_from_mesh

jax.config.update("jax_enable_x64", True)

MU, LMBDA = 1.0, 1.0


@autovmap(grad_u=2)
def strain_energy_density(grad_u):
    eps = 0.5 * (grad_u + grad_u.T)
    return MU * jnp.einsum("ij,ij->", eps, eps) + 0.5 * LMBDA * jnp.trace(eps) ** 2


def timed(fn, *args, repeats=5):
    """(first call, median of `repeats` later calls, cpu/wall of the later calls)"""
    t0 = time.perf_counter()
    jax.block_until_ready(fn(*args))
    first = time.perf_counter() - t0
    walls, cpus = [], []
    for _ in range(repeats):
        c0, t0 = time.process_time(), time.perf_counter()
        jax.block_until_ready(fn(*args))
        walls.append(time.perf_counter() - t0)
        cpus.append(time.process_time() - c0)
    return first, float(np.median(walls)), sum(cpus) / sum(walls)


def ms(t):
    return f"{1e3 * t:9.2f}"


print(f"jax {jax.__version__}, devices {jax.devices()}")
print(f"{'mesh':>9} {'n_dofs':>7} | {'compile [ms]: E / grad / hess':>32} | {'run [ms]: E / grad / hess':>32} | cpu/wall: E / grad / hess")
for n in (20, 50, 100, 200, 280):
    mesh = Mesh.unit_square(n, n)
    op = Operator(mesh, element.Tri3())
    n_dofs = 2 * mesh.coords.shape[0]

    def energy(u_flat):
        return op.integrate(strain_energy_density(op.grad(u_flat.reshape(-1, 2))))

    # the state used on both sides: u = 0.01 (sin(3x) y, x y²)
    x, y = mesh.coords[:, 0], mesh.coords[:, 1]
    u = 0.01 * jnp.stack([jnp.sin(3 * x) * y, x * y**2], axis=1).reshape(-1)

    cm = ColoredMatrix.from_csr(pattern_from_mesh(mesh, 2))
    n_colors = int(cm.colors.max()) + 1
    energy_fn = jax.jit(energy)
    grad_fn = jax.jit(jax.grad(energy))
    hess_fn = jax.jit(jacfwd(jax.grad(energy), cm, color_batch_size=n_colors))

    c_e, r_e, p_e = timed(energy_fn, u)
    c_g, r_g, p_g = timed(grad_fn, u)
    c_h, r_h, p_h = timed(lambda v: hess_fn(v).data, u)

    E, g, K = energy_fn(u), grad_fn(u), hess_fn(u).data
    print(
        f"{n:4d}x{n:<4d} {n_dofs:7d} | {ms(c_e)} {ms(c_g)} {ms(c_h)} | {ms(r_e)} {ms(r_g)} {ms(r_h)} | {p_e:4.1f} {p_g:4.1f} {p_h:4.1f}"
        f"   check: E={float(E):.12e} |g|={float(jnp.linalg.norm(g)):.12e} |K|={float(jnp.linalg.norm(K)):.12e} colors={n_colors}"
    )
