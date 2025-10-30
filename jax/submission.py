"""
Draft submission for a preconditioner which utilizes a randomized svd to compute preconditioned updates.
Before the preconditioner is applied, bias and weight terms are combined and reshaped to a matrix.
If the gradient is G, we first sketch G, then compute the SVD of the result, G ~ USV^T.
(In fact, here we sketch the momentum and compute the update based on the momentum).
Then we update the weights by UV^T.
If G is an m-by-n matrix, and d is the sketching dimension, this costs O(mnd).
The sketching and svd step can be sped up by sketching from the left and right, but the cost of reconstruction will still be O(mnd).
In the case where communication savings will be high enough, U, V can be computed in O(md^2) and O(nd^2) and then all reduced (with lower memory communication costs) and reconstructed after.
LayerNorm and BatchNorm parameters are optimized with nadamw.
"""

import functools
from typing import (
        Any,
        Callable,
        Dict,
        Iterator,
        List,
        NamedTuple,
        Optional,
        Tuple,
        Union,
        )

import chex
import jax
import jax.numpy as jnp
import optax

from . import linalg

from algoperf import spec, jax_sharding_utils

# parameter names -- 'scale','bias' -- LayerNorm/BatchNorm (Apply nadamw)
# 'kernel', 'embedding_table', 'embedding' -- higher-dimensional tensors to apply low rank orthogonal updates

HPARAMS = {
        'beta1': 0.9,           # momentum parameter for orthogonal updates and nadamw
        'beta2': 0.999,         # parameter for nadamw only (for the second moment)
        'krylov_iter': 2,       # number of iterations to use for finding range of input to svd
        'learning_rate': 0.01,  # learning rate
        'eps': 1e-8,            # eps value for nadamw 
        'eps_root': 0.0,        # sqrt(eps) value for nadamw
        'weight_decay': 0.01,   # weight_decay
        'dropout_rate': 0.1,    # dropout
        'rank_type': 'sqrt',    # or 'constant', decides what dimension to sketch
        'rank': None        # if 'rank_type'='constant', then what rank to use
        }

_GRAD_CLIP_EPS = 1e-6

class AugmentedShapeInfo(NamedTuple):
    """
    Class to keep track of transformations between 'kernel', 'bias' pairs and augmented matrix
    """
    kernel_name: str        # name of kernel weight
    kernel_shape: tuple     # shape of original kernel tensor
    bias_name: str          # name of bias weight
    bias_shape: tuple       # shape of original bias tensor
    reshaped_2d: tuple      # shape of kernel reshaped as matrix
    maybe_transpose: bool   # whether the augmented matrix should be transposed
    augmented_shape: tuple  # final shape of augmented matrix (after transposing)

def _is_shape_info(x):
    return isinstance(x, AugmentedShapeInfo)

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
        return weight_shape

    first_dim = weight_shape[0]
    rest_dim = int(jnp.prod(jnp.array(weight_shape[1:])))
    return (first_dim, rest_dim)

def _compute_shape_info(params):
    """
    Compute shape info for all parameters corresponding to
    'kernel', 'embedding' or 'embedding_table'
    In other words, all 2 or higher dimensional tensors that are parameters
    of the model.
    If the 'kernel', 'embedding' or 'embedding_table' is found, then it is combined
    with its 'bias', if there is a corresponding 'bias'.
    """

    def classify_subtree(subtree):
        if isinstance(subtree, dict) and (
                'kernel' in subtree or
                'embedding' in subtree or
                'embedding_table' in subtree
                ):
            if 'bias' in subtree:
                bias = subtree['bias']
                bias_name = 'bias'
                bias_shape = bias.shape
            else:
                bias = None
                bias_name = None
                bias_shape = None
            if 'kernel' in subtree:
                kernel_name = 'kernel'
            elif 'embedding' in subtree:
                kernel_name = 'embedding'
            elif 'embedding_table' in subtree:
                kernel_name = 'embedding_table'
            kernel = subtree[kernel_name]
            kernel_shape = kernel.shape

            reshaped_2d = _reshape_to_2d(kernel_shape, bias_shape)
            m, n = reshaped_2d
            maybe_transpose = m < n
            augmented_shape = (n + 1, m) if maybe_transpose else (m, n + 1)

            return AugmentedShapeInfo(
                    kernel_shape=kernel_shape,
                    kernel_name=kernel_name,
                    bias_shape=bias_shape,
                    bias_name=bias_name,
                    reshaped_2d=reshaped_2d,
                    maybe_transpose=maybe_transpose,
                    augmented_shape=augmented_shape
                    )
        else:
            return None

    return jax.tree.map(
            classify_subtree,
            params,
            is_leaf=lambda x: isinstance(x, dict) and (
                'kernel' in x or
                'embedding' in x or
                'embedding_table' in x
                )
            )

def create_param_labels() -> Callable:
    """
    Label the leaves of a Pytree according to whether the parameter
    should be an "Adam" parameter, or updated using low rank orthogonal updates.

    Returns a function that labels each parameter in a pytree as either
    - 'adam': for parameters of Layernorm or Batchnorm
    - 'low_rank_orthogonal_update': for weight matrices paired with their bias

    Returns:
        Callable: Function that takes params and returns labelled Pytree
    """
    def param_labels(params):
        shape_info = _compute_shape_info(params)

        def label_param(info):
            if info is None:
                return 'adam'
            else:
                return 'low_rank_orthogonal_update'

        return jax.tree.map(
                label_param,
                shape_info,
                is_leaf=lambda x: x is None or _is_shape_info(x)
                )
    return param_labels

def create_transform_shapes():
    """
    Create a transform `optimizer' which transforms updates into their augmented matrix shape
    """

    def init_fn(params):
        shape_info = _compute_shape_info(params)
        return shape_info

    def update_fn(updates, state, params=None):
        del params
        shape_info = state

        def augment_update(upd, info):
            kernel_upd = upd[info.kernel_name]
            kernel_2d = kernel_upd.reshape(info.reshaped_2d)
            if info.bias_name is not None:
                bias_row = upd[info.bias_name]
                bias_2d = bias_row.reshape(1, -1)
                augment_kernel = jnp.concatenate([kernel_2d, bias_2d], axis=0)
            else:
                augment_kernel = kernel_2d
            return augment_kernel
        transformed = jax.tree.map(
                augment_update,
                updates,
                shape_info,
                is_leaf=lambda x: isinstance(x, dict) or x is None or _is_shape_info(x)
                )
        return transformed, state
    return optax.GradientTransformation(init_fn, update_fn)

def create_inv_transform_shapes():
    """
    Create inverse function to function created by create_transform_shapes()
    """

    def init_fn(params):
        shape_info = _compute_shape_info(params)
        return shape_info

    def update_fn(updates, state, params=None):
        del params
        shape_info = state

        def unaugment_update(upd, info):

            m, n = info.reshaped_2d
            res_dict = {}
            if info.maybe_transpose:
                upd = upd.T
            weight_2d = upd[:m]
            res_dict[info.kernel_name] = weight_2d.reshape(info.kernel_shape)
            if info.bias_shape is not None:
                res_dict[info.bias_name] = upd[-1].flatten()
            return res_dict
        untransformed = jax.tree.map(
                unaugment_update,
                updates,
                shape_info,
                is_leaf=lambda x: x is None or _is_shape_info(x)
                )
        return untransformed, state
    return optax.GradientTransformation(init_fn, update_fn)

def create_svd_args(
        params,
        key: int,
        rank_type: str,
        rank: Optional[int] = None,
        ):
    """
    Return key Pytree and rank Pytree
    """
    def compute_rank(param_shape):
        if len(param_shape) != 2:
            return None
        m, n = param_shape

        if rank_type == 'sqrt':
            rank = int(jnp.sqrt(min(m, n)))
        elif rank_type == 'constant':
            if rank is None:
                raise ValueError("rank must be specified when rank_type=='constant'")
            rank = min(rank, min(m, n))
        else:
            raise ValueError(f"Unknown rank_type: {rank_type}")
        rank = max(1, min(rank, min(m, n)))
        return rank

    _, treedef = jax.tree.flatten(params)
    keys = jax.random.split(key, treedef.num_leaves)
    key_tree = jax.tree.unflatten(treedef, keys)
    rank_tree = jax.tree.map(
            lambda p: compute_rank(p.shape) if hasattr(p, 'shape') else None,
            params
            )
    return key_tree, rank_tree

class ScaleByLowRankOrthogonalUpdateState(NamedTuple):
    """State for the Low Rank Orthogonal Update algorithm.
    """
    step: chex.Array          # number of steps
    momentum: Any             # Pytree storing momentum of parameter
    krylov_iter: chex.Array   # number of krylov iterations
    key: Any                  # random key Pytree
    rank: Any                 # Pytree containing rank parameter for low rank svd

def create_optimizer_sharding(optimizer_state, replicated, sharded):
    """
    Create sharding spec for optimizer

    Args:
        optimizer_state: The optimizer state structure
        replicated: Sharding spec for replicated data
        sharded: Sharding spec for batch sharded data

    Returns:
        Sharding spec sharding rng key across batches and replicating
        all other optimizer variables
    """
    def shard_optimizer_component(state_component):
        if isinstance(state_component, ScaleByLowRankOrthogonalUpdateState):
            return ScaleByLowRankOrthogonalUpdateState(
                    step=replicated,
                    momentum=jax.tree.map(lambda _: replicated, state_component.momentum),
                    krylov_iter=replicated,
                    key=jax.tree.map(lambda _: sharded, state_component.key),
                    rank=jax.tree.map(lambda _: replicated, state_component.rank)
                    )
        else:
            return jax.tree.map(lambda _: replicated, state_component)

    return jax.tree.map(
            shard_optimizer_component,
            optimizer_state,
            is_leaf=lambda x: (
                isinstance(x, ScaleByLowRankOrthogonalUpdateState) or
                (
                    hasattr(x, '_fields') and
                    not isinstance(x, ScaleByLowRankOrthogonalUpdateState)
                    )
                )
            )

def low_rank_orthogonal_update(
        lr,
        key,
        beta1,
        beta2,
        krylov_iter,
        rank_type,
        rank_val,
        eps=1e-8,
        eps_root=0.0,
        weight_decay=0.0,
        mask=None):
    r"""

    Args:

    Returns:
        The corresponding `GradientTransformation`.
    """
    param_labels = create_param_labels()
    transform_shapes = create_transform_shapes()
    inv_transform_shapes = create_inv_transform_shapes()
    return optax.partition(
            transforms={
                'low_rank_orthogonal_update': optax.chain(
                    transform_shapes,
                    scale_by_low_rank_orthogonal_update(
                        key=key,
                        beta1=beta1,
                        krylov_iter=krylov_iter,
                        rank_type=rank_type,
                        rank_val=rank_val,
                        eps=eps
                        ),
                    inv_transform_shapes,
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
            param_labels=param_labels
            )




def scale_by_low_rank_orthogonal_update(
      key: int,
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
        momentum = jax.tree.map(
                jnp.zeros_like, params
                )
        key, rank = create_svd_args(params, key, rank_type, rank)
        return ScaleByLowRankOrthogonalUpdateState(
                step=jnp.zeros([], jnp.int32),
                momentum=momentum,
                krylov_iter=krylov_iter,
                rank=rank,
                key=key
                )

    def update_fn(updates, state, params=None):
        del params
        step_inc = state.step + 1
        momentum = _update_moment(updates, state.momentum, beta1, 1)
        key, consumable_key = jax.tree.map(jax.random.split, state.key)
        low_rank_momentum = jax.tree.map(
                linalg.svd_lowrank,
                momentum,
                consumable_key,
                state.rank,
                state.krylov_iter
                )
        return low_rank_momentum, ScaleByLowRankOrthogonalUpdateState(
                step=step_inc,
                momentum=momentum,
                key=key,
                krylov_iter=krylov_iter,
                rank=state.rank
                )
    return optax.GradientTransformation(init_fn, update_fn)



def _update_moment(updates, moments, decay, order):
    """Compute the exponential moving average of the `order`-th moment."""
    return jax.tree.map(
          lambda g, t: (1 - decay) * (g ** order) + decay * t, updates, moments)


def train_step(workload,
             opt_update_fn,
             model_state,
             optimizer_state,
             current_param_container,
             batch,
             rng,
             grad_clip,
             label_smoothing,
             dropout_rate,
             ):

    def _loss_fn(params):
        logits, new_model_state = workload.model_fn(
                params,
                batch,
                model_state,
                spec.ForwardPassMode.TRAIN,
                rng,
                update_batch_norm=True,
                dropout_rate=dropout_rate,
                )
        loss_dict = workload.loss_fn(
                label_batch=batch['targets'],
                logits_batch=logits,
                mask_batch=batch.get('weights'),
                label_smoothing=label_smoothing)
        summed_loss = loss_dict['summed']
        n_valid_examples = loss_dict['n_valid_examples']
        return summed_loss, (n_valid_examples, new_model_state)

    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
    (summed_loss, (n_valid_examples, new_model_state)), grad = grad_fn(
          current_param_container)
    # Get correct global mean loss and grad.
    loss = summed_loss / n_valid_examples
    grad = jax.tree.map(lambda x: x / n_valid_examples, grad)

    grad_norm = jnp.sqrt(
          sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grad)))

    if grad_clip is not None:
        grad_scaling_factor = grad_clip / (grad_norm + _GRAD_CLIP_EPS)
    grad_scaling_factor = jax.lax.clamp(min=0.0, x=grad_scaling_factor, max=1.0)
    grad = jax.tree.map(lambda x: x * grad_scaling_factor, grad)

    updates, new_optimizer_state = opt_update_fn(grad, optimizer_state,
                                               current_param_container)
    updated_params = optax.apply_updates(current_param_container, updates)
    return new_optimizer_state, updated_params, new_model_state, loss, grad_norm


def update_params(
        workload: spec.Workload,
        current_param_container: spec.ParameterContainer,
        current_params_types: spec.ParameterTypeTree,
        model_state: spec.ModelAuxiliaryState,
        hyperparameters: spec.Hyperparameters,
        batch: Dict[str, spec.Tensor],
        loss_type: spec.LossType,
        optimizer_state: spec.OptimizerState,
        eval_results: List[Tuple[int, float]],
        global_step: int,
        rng: spec.RandomState,
        train_state: Optional[Dict[str, Any]] = None) -> spec.UpdateReturn:
    """Return (updated_optimizer_state, updated_params, updated_model_state)."""
    del current_params_types
    del loss_type
    del train_state
    del eval_results
    del hyperparameters

    hyperparameters = HPARAMS

    optimizer_state, opt_update_fn = optimizer_state
    if 'label_smoothing' in hyperparameters:
        label_smoothing = hyperparameters['label_smoothing']
    else:
        label_smoothing = 0.0
    if 'grad_clip' in hyperparameters:
        grad_clip = hyperparameters['grad_clip']
    else:
        grad_clip = None
    dropout_rate = hyperparameters['dropout_rate']

    # mesh = jax.sharding.Mesh(jax.devices(), ('batch'))
    replicated = jax_sharding_utils.get_replicate_sharding()
    sharded = (
            jax_sharding_utils.get_batch_dim_sharding()
            )
    optimizer_sharding_spec = create_optimizer_sharding(
            optimizer_state, replicated=replicated, sharded=sharded
            )


    arg_shardings = (
            replicated, #model_state
            optimizer_sharding_spec, #optimizer_state # change to optimizer sharding eventually
            replicated, # current_param_container
            sharded, # batch
            replicated, # per_device_rngs
            replicated, # grad_clip
            replicated, #label_smoothing
            replicated, #dropout_rate
            )
    out_shardings = (
            optimizer_sharding_spec, # new_optimizer_state # maybe sharded eventually
            replicated, # updated_params
            replicated, # new_model_state
            replicated, # loss
            replicated, # grad_norm
            )
    jitted_train_step = jax.jit(
            train_step,
            static_argnums=(0, 1),
            donate_argnums=(2, 3, 4),
            in_shardings=arg_shardings,
            out_shardings=out_shardings,
            )
    outputs = jitted_train_step(workload,
                                opt_update_fn,
                                model_state,
                                optimizer_state,
                                current_param_container,
                                batch,
                                rng,
                                grad_clip,
                                label_smoothing,
                                dropout_rate,
                                )
    new_optimizer_state, new_params, new_model_state, loss, grad_norm = outputs

    # Log loss, grad_norm.
    if global_step % 100 == 0 and workload.metrics_logger is not None:
        workload.metrics_logger.append_scalar_metrics(
                {
                    'loss': loss,
                    'grad_norm': grad_norm,
                    }, global_step)
    return (new_optimizer_state, opt_update_fn), new_params, new_model_state



def prepare_for_eval(
        workload: spec.Workload,
        current_param_container: spec.ParameterContainer,
        current_params_types: spec.ParameterTypeTree,
        model_state: spec.ModelAuxiliaryState,
        hyperparameters: spec.Hyperparameters,
        loss_type: spec.LossType,
        optimizer_state: spec.OptimizerState,
        eval_results: List[Tuple[int, float]],
        global_step: int,
        rng: spec.RandomState,
        ) -> spec.UpdateReturn:
    del workload
    del hyperparameters
    del current_params_types
    del loss_type
    del eval_results
    del global_step
    del rng
    return (optimizer_state, current_param_container, model_state)

def get_batch_size(workload_name):
    if workload_name == 'criteo1tb':
        return 262_144
    elif workload_name == 'fastmri':
        return 32
    elif workload_name == 'imagenet_resnet':
        return 1024
    elif workload_name == 'imagenet_resnet_silu':
        return 512
    elif workload_name == 'imagenet_resnet_gelu':
        return 512
    elif workload_name == 'imagenet_vit':
        return 1024
    elif workload_name == 'librispeech_conformer':
        return 256
    elif workload_name == 'librispeech_deepspeech':
        return 256
    elif workload_name == 'ogbg':
        return 512
    elif workload_name == 'wmt':
        return 128
    elif workload_name == 'mnist':
        return 16
    elif workload_name == 'cifar':
        return 128
    else:
        raise ValueError(f'Unsupported workload name: {workload_name}.')

def data_selection(
        workload: spec.Workload,
        input_queue: Iterator[Dict[str, spec.Tensor]],
        optimizer_state: spec.OptimizerState,
        current_param_container: spec.ParameterContainer,
        model_state: spec.ModelAuxiliaryState,
        hyperparameters: spec.Hyperparameters,
        global_step: int,
        rng: spec.RandomState,
        ) -> Dict[str, spec.Tensor]:
    del workload
    del optimizer_state
    del current_param_container
    del model_state
    del hyperparameters
    del global_step
    del rng
    batch = next(input_queue)
    return batch

def init_optimizer_state(
        workload: spec.Workload,
        model_params: spec.ParameterContainer,
        model_state: spec.ModelAuxiliaryState,
        hyperparameters: spec.Hyperparameters,
        rng: spec.RandomState,
        ) -> spec.OptimizerState:
    del model_params
    del model_state
    params_zeros_like = jax.tree.map(
            lambda s: jnp.zeros(s.shape_tuple), workload.param_shapes
            )
    lr = HPARAMS['learning_rate']
    beta1 = HPARAMS['beta1']
    beta2 = HPARAMS['beta2']
    weight_decay = HPARAMS['weight_decay']
    krylov_iter = HPARAMS['krylov_iter']
    rank_type = HPARAMS['rank_type']  # 'sqrt' or 'constant'
    if rank_type == 'constant':
        rank_val = HPARAMS['rank']
    else:
        rank_val = None


    opt_init_fn, opt_update_fn = low_rank_orthogonal_update(
            key=rng,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            krylov_iter=krylov_iter,
            rank_type=rank_type,
            rank_val=rank_val
            )
    optimizer_state = opt_init_fn(params_zeros_like)
    return optimizer_state, opt_update_fn
