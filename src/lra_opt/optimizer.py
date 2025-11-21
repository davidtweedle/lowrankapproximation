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
        )

import collections
import chex
import jax
import jax.numpy as jnp
import optax

from absl import logging

import collections.abc

from flax import struct

from . import linalg


# parameter names -- 'scale','bias' -- LayerNorm/BatchNorm (Apply nadamw)
# 'kernel', 'embedding_table', 'embedding' -- higher-dimensional tensors to apply low rank orthogonal updates



#@struct.dataclass
#class AugmentedShapeInfo:
#    """
#    Class to keep track of transformations between 'kernel', 'bias' pairs and augmented matrix
#    """
#    kernel_name: str            = struct.field(pytree_node=False)    # name of kernel weight
#    kernel_shape: tuple         = struct.field(pytree_node=False)    # shape of original kernel tensor
#    bias_name: Optional[str]    = struct.field(pytree_node=False)    # name of bias weight
#    bias_shape: Optional[tuple] = struct.field(pytree_node=False)     # shape of original bias tensor
#    reshaped_2d: tuple          = struct.field(pytree_node=False)# shape of kernel reshaped as matrix
#    augmented_shape: tuple      = struct.field(pytree_node=False)# final shape of augmented matrix (after transposing)
#    factor_type: str            = struct.field(pytree_node=False)    # can be 'wide', 'tall', 'qr_with_pivot'
#    dtype: jnp.dtype            = struct.field(pytree_node=False)
#
#def _is_shape_info(x):
#    return isinstance(x, AugmentedShapeInfo)
#
#def _is_weight_block(x):
##    x = _get_raw_array(x)
##    if not hasattr(x, 'shape') or not hasattr(x, 'dtype'):
#        return False
#    if not jnp.issubdtype(x.dtype, jnp.floating):
#        return False
#    if len(x.shape) < 2:
        return False
#    return False
#
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

#def _compute_shape_info(params):
#    """
#    Compute shape info for all parameters corresponding to
#    'kernel', 'embedding' or 'embedding_table'
#    In other words, all 2 or higher dimensional tensors that are parameters
#    of the model.
#    If the 'kernel', 'embedding' or 'embedding_table' is found, then it is combined
#    with its 'bias', if there is a corresponding 'bias'.
#    """
#
#    def classify_subtree(subtree):
#        if _is_weight_block(subtree):
#            if isinstance(subtree, collections.abc.Mapping):
#                get_param = lambda key: subtree[key]
#                has_param = lambda key: key in subtree
#                all_keys = subtree.keys()
#            else:
#                get_param = lambda key: getattr(subtree, key)
#                has_param = lambda key: hasattr(subtree, key) and getattr(subtree, key) is not None
#                all_keys = dir(subtree)
#            if has_param('weight'):
#                kernel_name = 'weight'
#            elif has_param('kernel'):
#                kernel_name = 'kernel'
#            elif has_param('embedding'):
##                kernel_name = 'embedding'
##            elif has_param('embedding_table'):
##                kernel_name = 'embedding_table'
#            elif has_param('lm_head'):
#                kernel_name = 'lm_head'
#            else:
#                return None
#
#            if has_param('bias'):
#                bias_wrapper = get_param('bias')
#                if bias_wrapper is None:
##                    bias_name = None
#                    bias_shape = None
#                else:
#                    bias_name = 'bias'
##                    bias_raw = _get_raw_array(bias_wrapper)
#                    bias_shape = bias_raw.shape
#            else:
#                bias = None
#                bias_name = None
#                bias_shape = None
#            try:
#                kernel_wrapper = get_param(kernel_name)
#                kernel_raw = _get_raw_array(kernel_wrapper)
#                kernel_shape = kernel_raw.shape
#                dtype = kernel_raw.dtype
#            except AttributeError:
#                return None
#
#            reshaped_2d = _reshape_to_2d(kernel_shape, bias_shape)
#            m, n = reshaped_2d
#            extra = 1 if bias_name is not None else 0
#            augmented_shape = (m + extra, n)
#            if kernel_name in {'embedding', 'embedding_table'}:
#                factor_type = 'qr_with_pivot'
#            elif kernel_name == 'lm_head':
#                factor_type = 'qr_with_pivot_transpose'
#            elif m + extra >= n:
#                factor_type = 'tall'
#            else:
#                factor_type = 'wide'
#
#            return AugmentedShapeInfo(
#                    kernel_shape=kernel_shape,
#                    kernel_name=kernel_name,
#                    bias_shape=bias_shape,
#                    bias_name=bias_name,
#                    reshaped_2d=reshaped_2d,
#                    augmented_shape=augmented_shape,
#                    factor_type=factor_type,
#                    dtype=dtype,
#                    )
#        else:
#            return None
#
#    return jax.tree.map(
#            classify_subtree,
#            params,
#            is_leaf=_is_weight_block
#            )

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
    valid_indices = set()
    for i, (path, leaf) in enumerate(leaves):
        if isinstance(leaf, optax.MaskedNode):
            continue
        raw = _get_raw_array(leaf)
        if not hasattr(raw, 'shape') or not hasattr(raw, 'dtype'):
            continue
        path_to_info[path] = (i, raw)
        valid_indices.add(i)

    groups: List[ParameterGroup] = []
    consumed_indices = set()

    for path, (idx, raw) in path_to_info.items():
        if idx in consumed_indices:
            continue
        name = _get_path_name(path[-1])
        is_weight_name = name in ('kernel', 'weight', 'embedding',
                                  'embedding_table', 'lm_head', 'weights')
        is_float = jnp.issubdtype(raw.dtype, jnp.floating)
        is_matrix = len(raw.shape) >= 2
        if is_weight_name and is_float and is_matrix:
            prefix = path[:-1]
            last_node = path[-1]
            if isinstance(last_node, jax.tree_util.GetAttrKey):
                bias_node = jax.tree_util.GetAttrKey('bias')
            else:
                bias_node = jax.tree_util.DictKey('bias')
            bias_path = prefix + (bias_node,)
            bias_idx = None
            bias_shape = None
            if bias_path in path_to_info:
                b_idx, b_raw = path_to_info[bias_path]
                if len(b_raw.shape) == 1 and jnp.issubdtype(b_raw.dtype, jnp.floating):
                    bias_idx = b_idx
                    bias_shape = b_raw.shape
                    consumed_indices.add(b_idx)
            reshaped_2d = _reshape_to_2d(raw.shape)
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
    for b_name, bucket in final_buckets.items():
        for g_idx, group in enumerate(bucket.groups):
            leaf_to_bucket[group.weight_leaf_idx] = (b_name, g_idx, 'weight')
            if group.bias_leaf_idx is not None:
                leaf_to_bucket[group.bias_leaf_idx] = (b_name, g_idx, 'bias')

    logging.info("\n--- Optimizer Bucket Initialization Log ---")
    total_layers = sum(len(b.layer_paths) for b in final_map.values())
    total_unique_shapes = len(final_map)
    logging.info(f"Total Layers Partitioned: {total_layers}")
    logging.info(f"Total Unique Buckets: {total_unique_shapes}")
    for name, bucket in final_map.items():
        num_layers = len(bucket.layer_paths)
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


#def compute_bucket_structure(shape_info_tree: Any, params: Any, rank_type: str, rank_val: int = None) -> Dict[str, LayerBucket]:
#    leaves_with_paths, treedef = jax.tree.flatten_with_path(shape_info_tree, is_leaf=_is_shape_info)
#    valid_layers = []
#    for path, leaf in leaves_with_paths:
#        if isinstance(leaf, AugmentedShapeInfo):
#            valid_layers.append((path, leaf))
#    buckets_by_shape = collections.defaultdict(list)
#    special_bucket_map = {}
#    for path, info in valid_layers:
#        is_special_layer = info.kernel_name in {'embedding', 'embedding_table', 'lm_head'}
#        M, N = info.augmented_shape
#        if is_special_layer:
#            bucket_name = f"Special_{info.kernel_name}_{M}x{N}_idx{len(initial_bucket_map)}"
#            m, n = info.reshaped_2d
#            rank = _pick_rank(m, n, info.factor_type, rank_type, rank_val)
#            special_bucket_map[bucket_name] = LayerBucket(
#                    max_m=M,
#                    max_n=N,
#                    dtype=info.dtype,
#                    layer_paths=[path],
#                    shape_infos=[info],
#                    factor_type=info.factor_type,
#                    rank=rank
#                    )
#        else:
#            buckets_by_shape[(M, N, info.factor_type, info.dtype)].append((path, info))
#    initial_bucket_map = {}
#
#    for (M, N, factor_type, dtype), layer_list in buckets_by_shape.items():
#        bucket_name = f"Bucket_{M}x{N}"
#        max_m, max_n = M, N
#        layer_paths = [item[0] for item in layer_list]
#        shape_infos = [item[1] for item in layer_list]
#
#        initial_bucket_map[bucket_name] = LayerBucket(
#                max_m=max_m,
#                max_n=max_n,
#                layer_paths=layer_paths,
#                shape_infos=shape_infos,
#                dtype=dtype,
#                factor_type=factor_type,
#                )
#    merged_bucket_map = _merge_buckets(initial_bucket_map)
#    for name, bucket in merged_bucket_map.items():
#        m, n = bucket.max_m, bucket.max_n
#        factor_type = bucket.factor_type
#        merged_bucket_map[name] = bucket._replace(rank=_pick_rank(m, n, factor_type, rank_type, rank_val))
#    final_map = merged_bucket_map | special_bucket_map
#    logging.info("\n--- Optimizer Bucket Initialization Log ---")
#    total_layers = sum(len(b.layer_paths) for b in final_map.values())
#    total_unique_shapes = len(final_map)
#    logging.info(f"Total Layers Partitioned: {total_layers}")
#    logging.info(f"Total Unique Buckets: {total_unique_shapes}")
#    for name, bucket in final_map.items():
#        num_layers = len(bucket.layer_paths)
#        logging.info(f"\nBucket Name: {name} ({num_layers} layers)")
#        logging.info(f" Shape (M x N): {bucket.max_m} x {bucket.max_n}")
#        logging.info(f" Dtype: {bucket.dtype}")
#        logging.info(f" Factor type: {bucket.factor_type}")
#        logging.info(f" Total elements in Bucket Tensor: {num_layers * bucket.max_m * bucket.max_n}")
#    logging.info("---------------------")
#    return final_map

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
                new_total_area = (M_new * N_new) * (len(bucket_A.layer_paths) + len(bucket_B.layer_paths))
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
            bucket_merged = Layer_Bucket(
                    max_m=M_new,
                    max_n=N_new,
                    layer_paths=bucket_A.layer_paths + bucket_B.layer_paths,
                    shape_infos=bucket_A.shape_infos + bucket_B.shape_infos,
                    dtype=bucket_A.dtype
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

#def _get_leaf_by_key_path(tree: Any, path: Tuple) -> Any:
#    res = tree
#    for key_node in path:
#        if isinstance(key_node, (DictKey, FieldKey, GetAttrKey)):
#            key_val = key_node.key if hasattr(key_node, 'key') else key_node.name
#        elif isinstance(key_node, SequenceKey):
#            key_val = key_node.idx
#        else:
#            raise TypeError(f"Unknown JAX Key type in path: {type(key_node)}")
#        res = res[key_val]
#    return res

#def _tree_to_bucketed_tensors(
#        leaves: List[Any],
#        leaf_locs: List[Tuple],
#        bucket_structure: Dict[str, LayerBucket],
#        ) -> Dict[str, jnp.ndarray]:
#    bucket_lists = {
#            name: [None] * len(b.layer_paths)
#            for name, b in bucket_structure.items()
#            }
#    for leaf, loc in zip(leaves, leaf_locs):
#        if loc is None:
#            continue
#
#        if isinstance(leaf, collections.abc.Mapping):
#            get_param = lambda key: leaf[key]
#        else:
#            get_param = lambda key: getattr(leaf, key)
#
#        bucket_name, idx, info = loc
#        max_m = bucket_structure[bucket_name].max_m
#        max_n = bucket_structure[bucket_name].max_n
#
#        padded_matrix = jnp.zeros((max_m, max_n), dtype=info.dtype)
#        m, n = info.reshaped_2d
#        kernel_wrapper = get_param(info.kernel_name)
#        kernel_raw = _get_raw_array(kernel_wrapper)
#        kernel = kernel_raw.reshape(m, n)
#        padded_matrix = padded_matrix.at[:m, :n].set(kernel)
#
#        if info.bias_name is not None:
#            bias_wrapper = get_param(info.bias_name)
#            bias_raw = _get_raw_array(bias_wrapper)
#            bias_flat = bias_raw.reshape(-1)
#            padded_matrix = padded_matrix.at[m, :bias_flat.shape[0]].set(bias_flat)
#        bucket_lists[bucket_name][idx] = padded_matrix
#    batched_tensors = {
#            name: jnp.stack(mats) for name, mats in bucket_lists.items()
#            }
#    return batched_tensors
#
#    for bucket_name, bucket in bucket_structure.items():
#        M_max, N_max = bucket.max_m, bucket.max_n
#        num_layers = len(bucket.layer_paths)
#        dtype = bucket.dtype
#        bucket_tensor = jnp.zeros((num_layers, M_max, N_max), dtype=dtype)
#        for i, (path, info) in enumerate(zip(bucket.layer_paths, bucket.shape_infos)):
#            M_i, N_i = info.augmented_shape
#            padded_matrix = jnp.zeros((bucket.max_m, bucket.max_n), dtype=bucket.dtype)
#            layer = _get_leaf_by_key_path(updates, path)
#            m, n = info.reshaped_2d
#            kernel = layer[info.kernel_name].reshape(m, n)
#            padded_matrix = padded_matrix.at[:m, :n].set(kernel)
#            if info.bias_name is not None:
#                b2d = layer[info.bias_name].reshape(1, -1)
#                padded_matrix = padded_matrix.at[m: m + 1, :n].set(b2d)
#            bucket_tensor = bucket_tensor.at[i].set(padded_matrix)
#        batched_tensors[bucket_name] = bucket_tensor
#    return batched_tensors

#def _bucketed_tensors_to_tree(
#        batched_precond: Dict[str, jnp.ndarray],
#        leaf_locs: List[Optional[Tuple]],
#        treedef: Any,
#        original_leaves: List[Any],
#        ):  # spec.ParameterContainer?
#    new_leaves = []
#    for i, loc in enumerate(leaf_locs):
#        if loc is None:
#            new_leaves.append(original_leaves[i])
#            continue
#        bucket_name, idx, info = loc
#        precond_tensor = batched_precond[bucket_name][idx]
#        m, n = info.reshaped_2d
#        layer_precond_unpadded = precond_tensor[:m, :n]
#        layer_updates_dict = {}
#        layer_updates_dict[info.kernel_name] = layer_precond_unpadded.reshape(info.kernel_shape)
#        if info.bias_name is not None:
#            bias_len = math.prod(info.bias_shape)
#            layer_bias = precond_tensor[m, :bias_len]
#            layer_updates_dict[info.bias_name] = layer_bias.reshape(info.bias_shape)
#        new_leaves.append(layer_updates_dict)
#    return jax.tree.unflatten(treedef, new_leaves)
#    updates_flat = {} # {path_tuple: update_matrix,....}
#    for bucket_name, precond_tensor in batched_precond.items():
#        bucket = bucket_structure[bucket_name]
#        for i, (path, info) in enumerate(zip(bucket.layer_paths, bucket.shape_infos)):
#            layer_precond_padded = precond_tensor[i]
#            m, n = info.reshaped_2d
#            layer_precond_unpadded = layer_precond_padded[:m, :n]
#            layer_updates_dict = {}
#            layer_updates_dict[info.kernel_name] = layer_precond_unpadded.reshape(info.original_shape)
#            if info.bias_name is not None:
#                layer_bias = layer_precond_padded[m : m + 1, :n]
#                layer_updates_dict[info.bias_name] = layer_bias.reshape(info.original_bias_shape)
#
#            updates_flat[path] = layer_updates_dict
#    final_updates_tree = jax.tree.unflatten(updates_flat, original_template_tree)
#    return final_updates_tree


#def create_param_labels() -> Callable:
#    """
#    Label the leaves of a Pytree according to whether the parameter
#    should be an "Adam" parameter, or updated using low rank orthogonal updates.
#
#    Returns a function that labels each parameter in a pytree as either
#    - 'adam': for parameters of Layernorm or Batchnorm
#    - 'low_rank_orthogonal_update': for weight matrices paired with their bias
#
#    Returns:
#        Callable: Function that takes params and returns labelled Pytree
#    """
#    def param_labels(params):
#        shape_info = _compute_shape_info(params)
#
#        def label_param(info):
#            if info is None:
#                return 'adam'
#            else:
#                return 'low_rank_orthogonal_update'
#
#        return jax.tree.map(
#                label_param,
#                shape_info,
#                is_leaf=lambda x: x is None or _is_shape_info(x)
#                )
#    return param_labels

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
