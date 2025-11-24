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

def _get_reshape_info(shape: Tuple[int, ...], bias_size: Optional[int] = None) -> Tuple[Tuple[int, int], int]:
    total = math.prod(shape)
    if len(shape) < 2:
        return (total, 1,), 1

    if bias_size is not None:
        if shape[0] == bias_size:
            return (shape[0], total // shape[0]), 1
        elif shape[-1] == bias_size:
            return (total // shape[-1], shape[-1]), 0
        return (total // shape[-1], shape[-1]), 0
    m1, n1 = shape[0], total // shape[0]
    ratio1 = max(m1, n1) / min(m1, n1)

    m2, n2 = shape[-1], total // shape[-1]
    ratio2 = max(m2, n2) / min(m2, n2)

    if ratio1 < ratio2:
        return (m1, n1), 1
    else:
        return (m2, n2), 0


@struct.dataclass
class ParameterGroup:
    weight_leaf_idx: int = struct.field(pytree_node=False)
    bias_leaf_idx: Optional[int] = struct.field(pytree_node=False)

    weight_shape: Tuple[int, ...] = struct.field(pytree_node=False)
    bias_shape: Optional[Tuple[int, ...]] = struct.field(pytree_node=False)

    reshaped_2d: Tuple[int, int] = struct.field(pytree_node=False)
    augmented_shape: Tuple[int, int] = struct.field(pytree_node=False)
    concat_axis: int = struct.field(pytree_node=False)
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
        embedding_strategy='adam',
        lm_head_strategy='adam',
        ) -> Tuple[Dict[str, LayerBucket], Dict[int, Tuple[str, int, str]]]:
    leaves, treedef = jax.tree.flatten_with_path(params)
    attention_groups = collections.defaultdict(dict)
    other_leaves = {}
    biases = {}
    for i, (path, leaf) in enumerate(leaves):
        if isinstance(leaf, optax.MaskedNode):
            continue
        raw = _get_raw_array(leaf)
        if not hasattr(raw, 'shape') or not hasattr(raw, 'dtype'):
            continue
        effective_path = path
        if isinstance(path[-1], jax.tree_util.FlattenedIndexKey):
            effective_path = path[:-1]

        name = _get_path_name(effective_path[-1])
        parent_path = effective_path[:-1]
        if name == 'bias':
            biases[parent_path] = (i, raw)
        elif name in ('q_proj', 'k_proj', 'v_proj', 'query', 'key', 'value'):
            attention_groups[parent_path][name[0]] = (i, raw)
        elif len(raw.shape) >= 2:
            other_leaves[i] = (path, raw)

    groups: List[ParameterGroup] = []
    consumed_indices = set()
    for parent, members in attention_groups.items():
        if 'q' in members and 'k' in members and 'v' in members:
            q_idx, q_raw = members['q']
            k_idx, k_raw = members['k']
            v_idx, v_raw = members['v']

            q_path = leaves[q_idx][0]
            q_bias = biases.get(q_path[:-1])
            k_path = leaves[k_idx][0]
            k_bias = biases.get(k_path[:-1])
            v_path = leaves[v_idx][0]
            v_bias = biases.get(v_path[:-1])

            bias_indices = []
            if q_bias:
                bias_indices.append(q_bias[0])
                consumed_indices.add(q_bias[0])
            if k_bias:
                bias_indices.append(k_bias[0])
                consumed_indices.add(k_bias[0])
            if v_bias:
                bias_indices.append(v_bias[0])
                consumed_indices.add(v_bias[0])
            
            bias_size = q_bias[1].shape[0] if q_bias else None
            (base_m, base_n), concat_axis = _get_reshape_info(q_raw.shape, bias_size)
            if concat_axis == 0:
                fused_m, fused_n = base_m, base_n * 3
                aug_m, aug_n = fused_m + (1 if bias_indices else 0), fused_n
            else:
                fused_m, fused_n = base_m * 3, base_n
                aug_m, aug_n = fused_m, fused_n + (1 if bias_indices else 0)
            factor_type = 'tall' if aug_m >= aug_n else 'wide'
            groups.append(ParameterGroup(
                weight_leaf_idx=[q_idx, k_idx, v_idx],
                bias_leaf_idx=bias_indices if bias_indices else None,
                weight_shape=q_raw.shape,
                bias_shape=q_bias[1].shape if q_bias else None,
                reshaped_2d=(fused_m, fused_n),
                augmented_shape=(aug_m, aug_n),
                concat_axis=concat_axis,
                factor_type=f"{factor_type}_fused",
                dtype=q_raw.dtype,
                ))
            consumed_indices.update([q_idx, k_idx, v_idx])
        else:
            for idx, raw in members.values():
                other_leaves[idx] = (leaves[idx][0], raw)

    for idx, (path, raw) in sorted(other_leaves.items()):
        if idx in consumed_indices:
            continue
        name = _get_path_name(path[-1])
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
            (m, n), concat_axis = _get_reshape_info(raw.shape, bias_shape)
            aug_m, aug_n = m, n
            if bias_idx is not None:
                if concat_axis == 0:
                    aug_m += 1
                else:
                    aug_n += 1
            factor_type = None

            if name in {'embedding', 'embedding_table'} or (
                    len(raw.shape) == 2 and raw.shape[0] > 20000
                    ):
                if embedding_strategy == 'pivoted_qr':
                    factor_type = 'qr_with_pivot'
                else:
                    continue
            elif name == 'lm_head' or (
                    len(raw.shape) == 2 and raw.shape[1] > 20000
                    ):
                if lm_head_strategy == 'pivoted_qr':
                    factor_type = 'qr_with_pivot_transpose'
                else:
                    continue
            else:
                factor_type = 'tall' if aug_m >= aug_n else 'wide'

            group = ParameterGroup(
                    weight_leaf_idx=idx,
                    bias_leaf_idx=bias_idx,
                    weight_shape=raw.shape,
                    bias_shape=bias_shape,
                    reshaped_2d=(m, n),
                    augmented_shape=(aug_m, aug_n),
                    concat_axis=concat_axis,
                    factor_type=factor_type,
                    dtype=raw.dtype,
                    )
            groups.append(group)
            consumed_indices.add(idx)
    buckets_by_key = collections.defaultdict(list)
    special_bucket_map = {}
    for g in groups:
        if 'qr' in g.factor_type:
            name = f"Special_{g.augmented_shape[0]}x{g.augmented_shape[1]}_{id(g)}"
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
            w_idxs = group.weight_leaf_idx if isinstance(group.weight_leaf_idx, list) else [group.weight_leaf_idx]
            for w in w_idxs:
                leaf_to_bucket[w] = (b_name, g_idx, 'weight')
            if group.bias_leaf_idx is not None:
                b_idxs = group.bias_leaf_idx if isinstance(group.bias_leaf_idx, list) else [group.bias_leaf_idx]
                for b in b_idxs:
                    leaf_to_bucket[b] = (b_name, g_idx, 'bias')

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
        d = math.ceil(math.sqrt(max(2, int(n))))
        d = max(24, d)
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
    for b_name, bucket in bucket_structure.items():
        for g_idx, group in enumerate(bucket.groups):
            axis = group.concat_axis

            if isinstance(group.weight_leaf_idx, list):
                fusion_axis = 1 if axis == 0 else 0
                f_m, f_n = group.reshaped_2d
                single_shape = (f_m, f_n // 3) if fusion_axis == 1 else (f_m // 3, f_n)
                parts = [_get_raw_array(leaves[idx]).reshape(single_shape)
                         for idx in group.weight_leaf_idx]
                weight_mat = jnp.concatenate(parts, axis=fusion_axis)
            else:
                raw = _get_raw_array(leaves[group.weight_leaf_idx])
                weight_mat = raw.reshape(*group.reshaped_2d)
            if group.bias_leaf_idx is not None:
                if isinstance(group.bias_leaf_idx, list):
                    b_parts = [_get_raw_array(leaves[i]).reshape(-1) for i in group.bias_leaf_idx ]
                    b_vec = jnp.concatenate(b_parts, axis=0)
                else:
                    b_vec = _get_raw_array(leaves[group.bias_leaf_idx]).reshape(-1)
                current_m, current_n = weight_mat.shape
                if axis == 0:
                    bias_mat = b_vec.reshape(1, current_n)
                else:
                    bias_mat = b_vec.reshape(current_m, 1)
                final_mat = jnp.concatenate([weight_mat, bias_mat], axis=axis)
            else:
                final_mat = weight_mat
            mat = bucket_lists[b_name][g_idx]
            if mat is None:
                mat = jnp.zeros((bucket.max_m, bucket.max_n), dtype=bucket.dtype)
                bucket_lists[b_name][g_idx] = mat.at[:final_mat.shape[0], :final_mat.shape[1]].set(final_mat)

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
    new_leaves = list(original_leaves)
    for b_name, bucket in bucket_structure.items():
        update_batch = batched_updates[b_name]
        for g_idx, group in enumerate(bucket.groups):
            update_mat = update_batch[g_idx]

            valid_m, valid_n = group.augmented_shape
            valid_update = update_mat[:valid_m, :valid_n]
            axis = group.concat_axis

            if group.bias_leaf_idx is not None:
                if axis == 0:
                    weight_update = valid_update[:-1, :]
                    bias_update = valid_update[-1, :]
                else:
                    weight_update = valid_update[:, :-1]
                    bias_update = valid_update[:, -1]
            else:
                weight_update = valid_update
                bias_update = None
            if isinstance(group.weight_leaf_idx, list):
                fusion_axis = 1 if axis == 0 else 0
                dim_len = weight_update.shape[fusion_axis]
                chunk = dim_len // 3
                if fusion_axis == 0:
                    parts = [weight_update[i * chunk : (i + 1) * chunk, :] for i in range(3)]
                else:
                    parts = [weight_update[:, i * chunk : (i + 1) * chunk] for i in range(3) ]
                for i, idx in enumerate(group.weight_leaf_idx):
                    new_leaves[idx] = parts[i].reshape(*group.weight_shape)
            else:
                new_leaves[group.weight_leaf_idx] = weight_update.reshape(*group.weight_shape)
            if bias_update is not None:
                if isinstance(group.bias_leaf_idx, list):
                    total = bias_update.size
                    chunk = total // 3
                    b_parts = [bias_update[ i * chunk : (i + 1) * chunk] for i in range(3) ]
                    for i, idx in enumerate(group.bias_leaf_idx):
                        new_leaves[idx] = b_parts[i].reshape(group.bias_shape)
                else:
                    new_leaves[group.bias_leaf_idx] = bias_update.reshape(group.bias_shape)
    return jax.tree.unflatten(treedef, new_leaves)


def create_param_labels(embedding_strategy, lm_head_strategy) -> Callable:
    def label_fn(params):
        _, leaf_map = _analyze_tree_and_build_buckets(params, 'sqrt', None, embedding_strategy, lm_head_strategy)
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
        embedding_strategy='adam',
        lm_head_strategy='adam',
        adam_lr=None,
        eps=1e-8,
        eps_root=0.0,
        weight_decay=0.0,
        mask=None):
    r"""

    Args:

    Returns:
        The corresponding `GradientTransformation`.
    """
    param_label_fn = create_param_labels(embedding_strategy, lm_head_strategy)
    adam_learning_rate = adam_lr if adam_lr is not None else lr
    return optax.partition(
            transforms={
                'low_rank_orthogonal_update': optax.chain(
                    scale_by_low_rank_orthogonal_update(
                        key=key,
                        beta1=beta1,
                        krylov_iter=krylov_iter,
                        rank_type=rank_type,
                        rank_val=rank_val,
                        embedding_strategy=embedding_strategy,
                        lm_head_strategy=lm_head_strategy,
                        eps=eps
                        ),
                    optax.add_decayed_weights(weight_decay, mask),
                    optax.scale_by_learning_rate(lr)
                    ),
                'adam': optax.nadamw(
                    learning_rate=adam_learning_rate,
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
      embedding_strategy: str = 'adam',
      lm_head_strategy: str = 'adam',
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
                rank_val,
                embedding_strategy,
                lm_head_strategy,
                )
        _, treedef = jax.tree.flatten(params)
        batched_momentum = {
                name: jnp.zeros((len(b.groups), b.max_m, b.max_n), dtype=b.dtype)
                for name, b in bucket_structure.items()
                }
        if hasattr(jax.random, 'key_data'):
            key_raw = jax.random.key_data(key)
        else:
            key_raw = key
        return ScaleByLowRankOrthogonalUpdateState(
                step=jnp.zeros([], jnp.int32),
                bucket_structure=bucket_structure,
                momentum=batched_momentum,
                key=key_raw,
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
        rng_key = state.key
        if hasattr(jax.random, 'wrap_key_data'):
            rng_key = jax.random.wrap_key_data(state.key, impl='threefry2x32')

        num_buckets = len(state.bucket_structure)
        master_bucket_key, new_state_key_obj = jax.random.split(rng_key, 2)
        bucket_keys = jax.random.split(master_bucket_key, num_buckets)
        if hasattr(jax.random, 'key_data'):
            new_state_key = jax.random.key_data(new_state_key_obj)
        else:
            new_state_key = new_state_key_obj
        batched_momentum = state.momentum
        bucket_items = list(state.bucket_structure.items())
        batched_updated_momentum = {}
        for i, (name, bucket) in enumerate(bucket_items):
            use_key = bucket_keys[i]
            update = batched_updates[name]
            momentum = batched_momentum[name]
            rank = bucket.rank
            factor_type = bucket.factor_type.replace('_fused', '')
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
