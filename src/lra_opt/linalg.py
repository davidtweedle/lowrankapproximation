import jax
from jax import lax
import chex
import jax.numpy as jnp
import jax.scipy as jscp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt
from functools import partial

def generate_sparse_sign_embedding(
        key: jax.Array,
        d: int,
        m: int,
        k: int
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
    key_indices, key_signs = jax.random.split(key)
    indices = jax.vmap(
            lambda key: jax.random.choice(key, d, shape=(k,), replace=False))(jax.random.split(key_indices, m)).T
    signs = jax.random.choice(key_signs, jnp.array([-1, 1]), shape=(k, m))
    return indices, signs

def sparse_sign_embedding_matmul_kernel(
        input_ref,      # m-by-n input matrix
        indices_ref,    # k-by-m non-zero indices for each column
        signs_ref,      # k-by-m sign values for each column
        output_ref,     # d-by-n output matrix
        *,
        K: int,         # number of non-zeros per column
        BLOCK_SIZE_D, int,
        BLOCK_SIZE_N: int,
        ):
    pid_d = pl.program_id(axis=0)
    pid_n = pl.program_id(axis=1)

    d_start = pid_d * BLOCK_SIZE_D
    d_end = jnp.minimum(d_start + BLOCK_SIZE_D, output_ref.shape[0])
    n_start = pid_n * BLOCK_SIZE_N
    n_end = jnp.minimum(n_start + BLOCK_SIZE_N, output_ref.shape[1])

    m = input_ref.shape[0]

    for d_idx in range(d_start, d_end):
        for n_idx in range(n_start, n_end):
            acc = 0.0

            for col_idx in range(m):
                for k_idx in range(K):
                    row_idx = indices_ref[k_idx, col_idx]
                    if row_idx == d_idx:
                        sign = signs_ref[k_idx, col_idx]
                        acc += sign * input_ref[col_idx, n_idx]
                        break
            output_ref[d_idx, n_idx] = acc


def sparse_sign_embedding_matmul_optimized_kernel(
        input_ref,      # m-by-n input matrix
        indices_ref,    # k-by-m non-zero indices
        signs_ref,      # k-by-m signs
        output_ref,     # d-by-n output
        *,
        K: int,
        BLOCK_SIZE_N: int,
        BLOCK_SIZE_M: int
        ):
    pid_n = pl.program_id(axis=0)
    pid_m = pl.program_id(axis=1)

    n_start = pid_n * BLOCK_SIZE_N
    n_end = jnp.minimum(n_start + BLOCK_SIZE_N, input_ref.shape[1])
    m_start = pid_m * BLOCK_SIZE_M
    m_end = jnp.minimum(m_start + BLOCK_SIZE_M, input_ref.shape[0])

    for m_idx in range(m_start, m_end):
        col_indices = indices_ref[:, m_idx]
        col_signs = signs_ref[:, m_idx]

        for n_idx in range(n_start, n_end):
            input_val = input_ref[m_idx, n_idx]
            for k_idx in range(K):
                d_idx = col_indices[k_idx]
                sign = col_signs[k_idx]
                output_ref[d_idx, n_idx] += sign * input_val

def sparse_sign_embedding_multiply(
        input_matrix: jnp.ndarray,
        indices: jnp.ndarray,
        signs: jnp.ndarray,
        d: int,
        block_size_n: int = 32,
        block_size_m: int = 32,
        ) -> jnp.ndarray:
    m, n = input_matrix.shape
    k = indices.shape[0]

    grid = (
            (n + block_size_n - 1) // block_size_n,
            (m + block_size_m - 1) // block_size_m,
            )
    return pl.pallas_call(
            sparse_sign_embedding_matmul_optimized_kernel,
            out_shape=jax.ShapeDtypeStruct((d, n), input_matrix.dtype),
            grid=grid,
            in_specs=[
                pl.BlockSpec(lambda i, j: (j * block_size_m, i * block_size_n),
                             (block_size_m, block_size_n)),
                pl.BlockSpec(lambda i, j: (0, j * block_size_m),
                             (k, block_size_m)),
                pl.BlockSpec(lambda i, j: (0, j * block_size_m),
                             (k, block_size_m)),
                ],
            out_specs=pl.BlockSepc(memory_space=plt.SMEM),
            compiler_params={"K": k, "BLOCK_SIZE_N": block_size_n, "BLOCK_SIZE_M": block_size_m},
            )(input_matrix, indices, signs)
# forgot jax.jit?
def sparse_sign_embedding_matmul_xla(
        input_matrix: jnp.ndarray,
        indices: jnp.ndarray,
        signs: jnp.ndarray,
        d: int,
        ) -> jnp.ndarray:
    m, n = input_matrix.shape
    k = indices.shape[0]

    input_expanded = jnp.expand_dims(input_matrix, 0).repeat(k, axis=0)
    signs_expanded = signs[:, :, None]
    weighted_input = input_expanded * signs_expanded

    weighted_flat = weighted_input.reshape(k * m, n)
    indices_flat = indices.flatten()
    
    segment_ids = jnp.repeat(jnp.arange(m), k) * d + indices_flat

    output_flat = jax.ops.segment_sum(
            weighted_flat,
            segment_idx,
            num_segments=d * m,
            indices_are_sorted=False
            )
    output = output_flat.reshape(m, d, n).sum(axis=0)
    return output


@partial(jax.jit, static_argnames=['d'])
def srct_columnwise(
        x: jnp.ndarray,
        key: jax.random.PRNGKey,
        d: int,
        ) -> jnp.ndarray:
    # based on type II discrete cosine transform
    m, n = x.shape
    key_signs, key_subsample = jax.random.split(key)
    signs = jax.random.choice(key_signs, jnp.array([1.0, -1.0]), shape=(m, 1))
    x_flipped = x * signs

    x_padded = jnp.concatenate([x_signed, x_signed[::-1, :]], axis=0)
    x_fft = jnp.fft.fft(x_padded, axis=0)[:m, :]

    k = jnp.arange(m)[:, None]
    phase = jnp.exp(-1j * jnp.pi * k / (2 * m))
    x_dct = 2 * jnp.real(x_fft * phase)

    x_dct = x_dct * jnp.sqrt(1.0 / (2 * m))
    x_dct = x_dct.at[0, :].multiply(1.0 / jnp.sqrt(2))

    indices = jax.random.choice(key_subsample, m, shape=(d,), replace=False)
    x_subsampled = x_dct[indices, :]
    x_subsampled = x_subsampled * jnp.sqrt(m / d)

    return x_subsampled

@partial(jax.jit, static_argnums=(2,3,4))
def compute_update(
        x: jnp.ndarray,
        key: jax.Array,
        d: int,
        niter: int = 2,
        factor_type: str = 'tall',
        ) -> jnp.ndarray:
    if factor_type == 'tall':
        m, n = x.shape  # assume that m >= n

        Q = get_approximate_basis(x, key, d, niter, factor_type)
        # returns an orthogonal m-by-(niter + 1) * d matrix
        B = Q.T @ x
        Ub, _, Vh = jnp.linalg.svd(B, full_matrices=False)
        U = Q @ Ub
        update = U @ Vh
    elif factor_type == 'wide':
        m, n = x.shape
        Q = get_approximate_basis(x, key, d, niter, factor_type)
        B = x @ Q
        U, _, Vhb = jnp.linalg.svd(B.T, full_matrices=False)
        Vh = Vhb @ Q
        update = U @ Vh
    elif factor_type == 'right_side_only':
        m, n = x.shape
        Q = get_approximate_basis(x, key, d, niter, 'tall')
        B = Q.T @ x
        Ub, S, Vh = jnp.linalg.svd(B, full_matrices=False)
        U = Q @ Ub
        update = (U * jnp.sqrt(S)) @ Vh
    elif factor_type == 'qr_with_pivot':
        update = rowspace_update_from_sketch(
                x,
                key,
                d=d,
                power=niter,
                oversample=8
                )
    elif factor_type == 'qr_with_pivot_transpose':
        update = rowspace_update_from_sketch(
                x.T,
                key,
                d=d,
                power=niter,
                oversample=8
                ).T
    return update

def get_approximate_basis(
        x: jnp.ndarray,
        key: jax.Array,
        d: int,
        niter: int = 2,
        factor_type: str = 'tall',
        ) -> jnp.ndarray:
    '''
    Uses Krylov block power iteration
    Uses gaussian random matrix
    '''
    if factor_type == 'tall':
        m, n = x.shape
        R = jax.random.normal(key=key, shape=(n, d), dtype=x.dtype)
        Y = x @ R
        Q, _ = jnp.linalg.qr(Y, mode='reduced')

        def body(_, Qcur):
            Z = x.T @ Qcur
            Qt, _ = jnp.linalg.qr(Z, mode='reduced')
            Y = x @ Qt
            Qnew, _ = jnp.linalg.qr(Y, mode='reduced')
            return Qnew

        Q = lax.fori_loop(0, int(niter), body, Q)
    elif factor_type == 'wide':
        m, n = x.shape
        R = jax.random.normal(key=key, shape=(d, m), dtype=x.dtype)
        Y = R @ x
        Q, _ = jnp.linalg.qr(Y.T, mode='reduced')

        def body(_, Qcur):
            Z = Qcur @ x
            Qt, _ = jnp.linalg.qr(Z, mode='reduced')
            Y = x.T @ Qt
            Qnew, _ = jnp.linalg.qr(Y, mode='reduced')
            return Qnew
        Q = lax.fori_loop(0, int(niter), body, Q)
    return Q

@partial(jax.jit, static_argnums=(2,3,4))
def rowspace_update_from_sketch(
        x: jnp.ndarray,
        key: jax.Array,
        d: int,
        power: int = 2,
        oversample: int = 8
        ) -> jnp.ndarray:

    m, n = x.shape
    d_eff = min(int(d) + int(oversample), m)
    G = jax.random.normal(key, (n, d_eff), dtype=x.dtype)
    Y = x @ G

    def body(_, Ycur):
        Z = x.T @ Ycur
        Qz, _ = jnp.linalg.qr(Z, mode='reduced')
        
        Ynew = x @ Qz
        Qy, _ = jnp.linalg.qr(Ynew, mode='reduced')
        return Qy

    Y = jax.lax.fori_loop(0, int(power), body, Y)

    Qt, Rt, P = jscp.linalg.qr(Y.T, mode='economic', pivoting=True)
    P_sel = P[:int(d)]
    Qr, _ = jnp.linalg.qr(x[P_sel, :].T, mode='reduced')
    update = jnp.zeros_like(x)
    update = update.at[P_sel, :].set(Qr.T)
    return update
