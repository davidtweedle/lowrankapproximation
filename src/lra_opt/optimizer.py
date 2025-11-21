"""
A preconditioner which utilizes a randomized svd to compute preconditioned updates.
Before the preconditioner is applied, bias and weight terms are combined and reshaped to a matrix.
If the gradient is G, we first sketch G, then compute the SVD of the result, G ~ USV^T.
(In fact, here we sketch the momentum and compute the update based on the momentum).
Then we update the weights by UV^T.
If G is an m-by-n matrix, and d is the sketching dimension, this costs O(mnd).
The sketching and svd step can be sped up by sketching from the left and right, but the cost of reconstruction will still be O(mnd).
In the case where communication savings will be high enough, U, V can be computed in O(md^2) and O(nd^2) and then all reduced (with lower memory communication costs) and reconstructed after.
LayerNorm and BatchNorm parameters are optimized with nadamw.
"""

import math
from typing import (
        Any,
        Callable,
        Dict,
        List,
        NamedTuple,
        Optional,
        Tuple,
        Union,
        )

import collections
import collections.abc
import chex
import jax
import jax.numpy as jnp
import optax
from absl import logging
from flax import struct


from . import linalg


def _get_raw_array(tensor):
    if hasattr(tensor, "array") and hasattr(tensor, "axes"):
        return tensor.array
    return tensor

def _reshape_to_2d(weight_shape, bias_shape) -> Tuple[int, int]:
    """
    Calculate shape of weight as a matrix
    Args:
        weight_shape: Shape tuple of weight tensor
        bias_shape: Shape tuple of bias tensor (unused, but kept for API consideration)

    Returns:
        tuple: (first_dim, product_of_remaining_dims) or original shape if matrix
    """
    if len(weight_shape) <= 2:
        return tuple(int(d) for d in weight_shape)
    
    if len(weight_shape) == 3:
        d0, d1, d2 = weight_shape
        if d0 > d1:  #QKV kernel
            return (int(d0), int(d1 * d2))
        else:  # out proj
            return (int(d0 * d1), int(d2))
    if len(weight_shape) > 3:
        out_dim = int(weight_shape[-1])
        in_dim = int(math.prod(int(d) for d in weight_shape[:-1]))
    return (in_dim, out_dim)

@struct.dataclass
class ParameterGroup:
    weight_leaf_idx: int = struct.field(pytree_node=False)
    bias_leaf_idx: Optional[int] = struct.field(pytree_node=False)

    weight_shape: Tuple[int, ...] = struct.field(pytree_node=False)
    bias_shape: Optional[Tuple[int, ...]] = struct.field(pytree_node=False)

    reshaped_2d: Tuple[int, int] = struct.field(pytree_node=False)
    augmented_shape: Tuple[int, int] = struct.field(pytree_node=False)
    factor_type: str = struct.field(pytree_node=False)
    dtype: Any = struct.field(pytree_node=False)


class LayerBucket(NamedTuple):
    max_m: int
    max_n: int
    rank: int
    groups: List[ParameterGroup]
    dtype: jnp.dtype
    factor_type: str

def _get_path_name(path_node) -> str:
    if hasattr(path_node, 'key'):
        return str(path_node.key)
    if hasattr(path_node, 'name'):
        return str(path_node.name)
    return str(path_node)

def _analyze_tree_and_build_buckets(
        params,
        rank_type,
        rank_val,
        ) -> Tuple[Dict[str, LayerBucket], Dict[int, Tuple[str, int, str]]]:
    leaves, treedef = jax.tree.flatten_with_path(params)
    path_to_info = {}
    for i, (path, leaf) in enumerate(leaves):
        if isinstance(leaf, optax.MaskedNode):
            continue
        raw = _get_raw_array(leaf)
        if not hasattr(raw, 'shape') or not hasattr(raw, 'dtype'):
            continue
        path_to_info[path] = (i, raw)

    groups: List[ParameterGroup] = []
    consumed_indices = set()

    for path, (idx, raw) in path_to_info.items():
        if idx in consumed_indices:
            continue
        effective_path = path
        if isinstance(path[-1], jax.tree_util.FlattenedIndexKey):
            effective_path = path[:-1]
        name = _get_path_name(effective_path[-1])
        is_weight_name = name in ('kernel', 'weight', 'embedding',
                                  'embedding_table', 'lm_head', 'weights')
        is_float = jnp.issubdtype(raw.dtype, jnp.floating)
        is_matrix = len(raw.shape) >= 2
        if is_weight_name and is_float and is_matrix:
            prefix = effective_path[:-1]
            last_key_node = effective_path[-1]
            if isinstance(last_key_node, jax.tree_util.GetAttrKey):
                bias_node = jax.tree_util.GetAttrKey('bias')
            else:
                bias_node = jax.tree_util.DictKey('bias')
            bias_path_base = prefix + (bias_node,)
            bias_idx = None
            bias_shape = None
            if bias_path_base in path_to_info:
                target_bias_path = bias_path_base
            elif (bias_path_base + (jax.tree_util.FlattenedIndexKey(0),)) in path_to_info:
                target_bias_path = bias_path_base + (jax.tree_util.FlattenedIndexKey(0),)
            else:
                target_bias_path = None
            if target_bias_path:
                b_idx, b_raw = path_to_info[target_bias_path]
                if jnp.issubdtype(b_raw.dtype, jnp.floating):
                    bias_idx = b_idx
                    bias_shape = b_raw.shape
                    consumed_indices.add(b_idx)
            reshaped_2d = _reshape_to_2d(raw.shape, bias_shape)
            m, n = reshaped_2d
            extra = 1 if bias_idx is not None else 0
            augmented_shape = (m + extra, n)

            factor_type = 'wide'
            if name in {'embedding', 'embedding_table'} or (
                    len(raw.shape) == 2 and raw.shape[0] > 20000
                    ):
                factor_type = 'qr_with_pivot'
            elif name == 'lm_head' or (
                    len(raw.shape) == 2 and raw.shape[1] > 20000
                    ):
                factor_type = 'qr_with_pivot_transpose'
            elif m + extra >= n:
                factor_type = 'tall'

            group = ParameterGroup(
                    weight_leaf_idx=idx,
                    bias_leaf_idx=bias_idx,
                    weight_shape=raw.shape,
                    bias_shape=bias_shape,
                    reshaped_2d=reshaped_2d,
                    augmented_shape=augmented_shape,
                    factor_type=factor_type,
                    dtype=raw.dtype,
                    )
            groups.append(group)
            consumed_indices.add(idx)
    buckets_by_key = collections.defaultdict(list)
    special_bucket_map = {}
    for g in groups:
        if g.factor_type.startswith('qr'):
            name = f"Special_{g.augmented_shape[0]}x{g.augmented_shape[1]}_{g.weight_leaf_idx}"
            rank = _pick_rank(*g.augmented_shape, g.factor_type, rank_type, rank_val)
            special_bucket_map[name] = LayerBucket(
                    max_m=g.augmented_shape[0],
                    max_n=g.augmented_shape[1],
                    rank=rank,
                    groups=[g],
                    dtype=g.dtype,
                    factor_type=g.factor_type,
                    )
        else:
            key = (g.augmented_shape, g.factor_type, g.dtype)
            buckets_by_key[key].append(g)
    final_buckets = {}
    for (shape, f_type, dtype), g_list in buckets_by_key.items():
        name = f"Bucket_{shape[0]}x{shape[1]}"
        rank = _pick_rank(shape[0], shape[1], f_type, rank_type, rank_val)
        final_buckets[name] = LayerBucket(
                max_m=shape[0],
                max_n=shape[1],
                rank=rank,
                groups=g_list,
                dtype=dtype,
                factor_type=f_type,
                )
    merged_buckets = _merge_buckets(final_buckets)
    merged_buckets.update(special_bucket_map)
    leaf_to_bucket = {}
    for b_name, bucket in merged_buckets.items():
        for g_idx, group in enumerate(bucket.groups):
            leaf_to_bucket[group.weight_leaf_idx] = (b_name, g_idx, 'weight')
            if group.bias_leaf_idx is not None:
                leaf_to_bucket[group.bias_leaf_idx] = (b_name, g_idx, 'bias')

    logging.info("\n--- Optimizer Bucket Initialization Log ---")
    total_layers = sum(len(b.groups) for b in merged_buckets.values())
    total_unique_shapes = len(merged_buckets)
    logging.info(f"Total Layers Partitioned: {total_layers}")
    logging.info(f"Total Unique Buckets: {total_unique_shapes}")
    for name, bucket in merged_buckets.items():
        num_layers = len(bucket.groups)
        logging.info(f"\nBucket Name: {name} ({num_layers} layers)")
        logging.info(f" Shape (M x N): {bucket.max_m} x {bucket.max_n}")
        logging.info(f" Dtype: {bucket.dtype}")
        logging.info(f" Factor type: {bucket.factor_type}")
        logging.info(f" Total elements in Bucket Tensor: {num_layers * bucket.max_m * bucket.max_n}")
    logging.info("---------------------")

    return merged_buckets, leaf_to_bucket


def _pick_rank(m, n, factor_type, rank_type, rank_val=None) -> Optional[int]:
    if factor_type == 'qr_with_pivot':
        d = math.ceil(10 * math.log2(max(2, int(n))))
        d = max(24, d)
        k = min(n, math.ceil(math.sqrt(int(n))))
        d = min(k, d)
        return int(d)
    rmax = min(int(m), int(n))
    if rank_type == 'sqrt':
        r = int(math.sqrt(rmax))
    elif rank_type == 'constant':
        if rank_val is None:
            raise ValueError("rank_val must be set for rank_type='constant'")
        r = int(min(rank_val, rmax))
    else:
        raise ValueError(f"Unknown rank_type: {rank_type}")
    r = max(1, r)
    r_po2 = 1 << (r - 1).bit_length()
    return r_po2


def _merge_buckets(initial_bucket_map):
    ActiveBuckets: List[Tuple(int, LayerBucket)] = [(bucket.max_m * bucket.max_n, bucket) for bucket in initial_bucket_map.values() ]
    ActiveBuckets.sort(key=lambda b: b[0])
    while True:
        best_cost = float('inf')
        best_pair = None
        N = len(ActiveBuckets)
        for i in range(N):
            for j in range(i + 1, N):

                area_A, bucket_A = ActiveBuckets[i]
                area_B, bucket_B = ActiveBuckets[j]
                M_new = max(bucket_A.max_m, bucket_B.max_m)
                N_new = max(bucket_A.max_n, bucket_B.max_n)
                new_total_area = (M_new * N_new) * (len(bucket_A.groups) + len(bucket_B.groups))
                total_useful_area = area_A + area_B
                padding_cost = (new_total_area - total_useful_area) / new_total_area
                if padding_cost < best_cost:
                    best_cost = padding_cost
                    best_pair = (i, j)
        tau_pad = 0.2

        if best_cost <= tau_pad:
            i, j = best_pair  # i < j
            area_A, bucket_A = ActiveBuckets[i]
            area_B, bucket_B = ActiveBuckets[j]

            M_new = max(bucket_A.max_m, bucket_B.max_m)
            N_new = max(bucket_A.max_n, bucket_B.max_n)
            bucket_merged = LayerBucket(
                    max_m=M_new,
                    max_n=N_new,
                    rank=max(bucket_A.rank, bucket_B.rank),
                    groups=bucket_A.groups + bucket_B.groups,
                    dtype=bucket_A.dtype,
                    factor_type=bucket_A.factor_type
                    )
            del ActiveBuckets[j]
            ActiveBuckets[i] = (area_A + area_B, bucket_merged)
        else:
            logging.info(f"Stopping merge: Best cost was {best_cost:.2f}k which is > {tau_pad}")
            break
    final_bucket_map = {}
    for _, bucket in ActiveBuckets:
        M = bucket.max_m
        N = bucket.max_n
        bucket_name = f"Bucket_{M}x{N}"
        final_bucket_map[bucket_name] = bucket
    return final_bucket_map


def _leaves_to_bucketed_tensors(
        leaves: List[Any],
        leaf_map: Dict[int, Tuple[str, int, str]],
        bucket_structure: Dict[str, LayerBucket],
        ) -> Dict[str, jnp.ndarray]:
    bucket_lists = {
            name: [None] * len(b.groups)
            for name, b in bucket_structure.items()
            }
    for leaf_idx, (b_name, g_idx, kind) in leaf_map.items():
        leaf = leaves[leaf_idx]
        if isinstance(leaf, optax.MaskedNode):
            continue
        leaf_raw = _get_raw_array(leaf)
        bucket = bucket_structure[b_name]
        group = bucket.groups[g_idx]

        mat = bucket_lists[b_name][g_idx]
        if mat is None:
            mat = jnp.zeros((bucket.max_m, bucket.max_n), dtype=bucket.dtype)
        m, n = group.reshaped_2d
        if kind == 'weight':
            kernel = leaf_raw.reshape(m, n)
            mat = mat.at[:m, :n].set(kernel)
        elif kind == 'bias':
            bias_flat = leaf_raw.reshape(-1)
            mat = mat.at[m, :bias_flat.shape[0]].set(bias_flat)
        bucket_lists[b_name][g_idx] = mat
    return {
            name: jnp.stack(mats)
            for name, mats in bucket_lists.items()
            }

def _bucketed_tensors_to_tree(
        batched_updates: Dict[str, jnp.ndarray],
        leaf_map: Dict[int, Tuple[str, int, str]],
        bucket_structure: Dict[str, LayerBucket],
        original_leaves: List[Any],
        treedef: Any,
        ):
    new_leaves = []
    for i, leaf in enumerate(original_leaves):
        if i not in leaf_map:
            new_leaves.append(leaf)
            continue
        b_name, g_idx, kind = leaf_map[i]
        bucket = bucket_structure[b_name]
        group = bucket.groups[g_idx]

        update_mat = batched_updates[b_name][g_idx]
        m, n = group.reshaped_2d
        if kind == 'weight':
            update = update_mat[:m, :n].reshape(group.weight_shape)
            new_leaves.append(update)
        elif kind == 'bias':
            update = update_mat[m, :n].reshape(group.bias_shape)
            new_leaves.append(update)
    return jax.tree.unflatten(treedef, new_leaves)


def create_param_labels() -> Callable:
    def label_fn(params):
        _, leaf_map = _analyze_tree_and_build_buckets(params, 'sqrt', None)
        leaves, treedef = jax.tree.flatten_with_path(params)
        labels = []
        for i, _ in enumerate(leaves):
            if i in leaf_map:
                labels.append('low_rank_orthogonal_update')
            else:
                labels.append('adam')
        return jax.tree.unflatten(treedef, labels)
    return label_fn


@struct.dataclass    
class ScaleByLowRankOrthogonalUpdateState:
    """State for the Low Rank Orthogonal Update algorithm.
    """
    step: chex.Array          # number of steps
    bucket_structure: Any = struct.field(pytree_node=False)
    momentum: Dict[str, jnp.ndarray]
    key: chex.Array
    leaf_map: Dict[int, Tuple[str, int, str]] = struct.field(pytree_node=False)
    treedef: Any = struct.field(pytree_node=False)


def low_rank_orthogonal_update(
        lr,
        key,
        beta1,
        beta2,
        krylov_iter,
        rank_type,
        rank_val,
        param_label_fn=None,
        eps=1e-8,
        eps_root=0.0,
        weight_decay=0.0,
        mask=None):
    r"""

    Args:

    Returns:
        The corresponding `GradientTransformation`.
    """
    if param_label_fn is None:
        param_label_fn = create_param_labels()
    return optax.partition(
            transforms={
                'low_rank_orthogonal_update': optax.chain(
                    scale_by_low_rank_orthogonal_update(
                        key=key,
                        beta1=beta1,
                        krylov_iter=krylov_iter,
                        rank_type=rank_type,
                        rank_val=rank_val,
                        eps=eps
                        ),
                    optax.add_decayed_weights(weight_decay, mask),
                    optax.scale_by_learning_rate(lr)
                    ),
                'adam': optax.nadamw(
                    learning_rate=lr,
                    b1=beta1,
                    b2=beta2,
                    eps=eps,
                    eps_root=eps_root,
                    weight_decay=weight_decay,
                    mask=mask
                    )
                },
            param_labels=param_label_fn
            )


def scale_by_low_rank_orthogonal_update(
      key: chex.Array,
      beta1: float = 0.9,
      krylov_iter: int = 2,
      rank_type: str = 'sqrt',
      rank_val: Optional[int] = None,
      eps=1e-8,
      ):
    """
    Scale gradients using low-rank SVD-based orthogonal updates

    This optimizer maintains momentum and applies preconditioned updates
    via randomized SVD.

    Args:
    key: Random key for sketching method
    beta1: Momentum parameter (default: 0.9)
    krylov_iter: Number of Krylov iterations for svd range finding (default: 2)
    rank_type: How to determine SVD rank - 'sqrt' or 'constant' (TODO: add 'log')
    rank: Fixed rank if rank_type='constant'
    eps: Small constant for numerical stability (default: 1e-8)

    Returns:
    GradientTransformation: Optax gradient transformation

    Note:
    Assumes params are matrices
    """
    def init_fn(params):
        # calculate buckets
        bucket_structure, leaf_map = _analyze_tree_and_build_buckets(
                params,
                rank_type,
                rank_val
                )
        _, treedef = jax.tree.flatten(params)
        batched_momentum = {
                name: jnp.zeros((len(b.groups), b.max_m, b.max_n), dtype=b.dtype)
                for name, b in bucket_structure.items()
                }
        return ScaleByLowRankOrthogonalUpdateState(
                step=jnp.zeros([], jnp.int32),
                bucket_structure=bucket_structure,
                momentum=batched_momentum,
                key=key,
                treedef=treedef,
                leaf_map=leaf_map,
                )

    def update_fn(updates, state, params=None):
        del params
        step_inc = state.step + 1
        update_leaves = jax.tree.leaves(updates)
        batched_updates = _leaves_to_bucketed_tensors(
                update_leaves,
                state.leaf_map,
                state.bucket_structure
                )

        num_buckets = len(state.bucket_structure)
        master_bucket_key, new_state_key = jax.random.split(state.key, 2)
        bucket_keys = jax.random.split(master_bucket_key, num_buckets)
        batched_momentum = state.momentum
        bucket_items = list(state.bucket_structure.items())
        batched_updated_momentum = {}
        for i, (name, bucket) in enumerate(bucket_items):
            use_key = bucket_keys[i]
            update = batched_updates[name]
            momentum = batched_momentum[name]
            rank = bucket.rank
            factor_type = bucket.factor_type
            updated_momentum = (1 - beta1) * update + beta1 * momentum
            batched_updated_momentum[name] = updated_momentum
            batched_updates[name] = linalg.compute_batched_update(
                    updated_momentum,
                    use_key,
                    rank,
                    krylov_iter,
                    factor_type,
                    )
        tree_updates = _bucketed_tensors_to_tree(
                batched_updates,
                state.leaf_map,
                state.bucket_structure,
                update_leaves,
                state.treedef,
                )

        new_state = ScaleByLowRankOrthogonalUpdateState(
                step=step_inc,
                momentum=batched_updated_momentum,
                key=new_state_key,
                bucket_structure=state.bucket_structure,
                treedef=state.treedef,
                leaf_map=state.leaf_map
                )
        return tree_updates, new_state
    return optax.GradientTransformation(init_fn, update_fn)
