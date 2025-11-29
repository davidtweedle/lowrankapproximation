import dataclasses
from dataclasses import dataclass, field
from functools import partial
from typing import Optional, Dict, Any, Union, List

import jax
import jax.numpy as jnp
import optax

from levanter.optim.config import LrSchedule, LrScheduleContext

from levanter.callbacks.watch import WatchConfig
from levanter.optim.muon import scale_with_muon
from haliax.nn import Linear
from levanter.utils.jax_utils import leaf_key_paths
from levanter.optim import OptimizerConfig
from levanter.schedule import IntSchedule

from marin.resources import GpuConfig, TpuPodConfig


@LrSchedule.register_subclass("wsd")
@dataclass(frozen=True)
class WSDLrSchedule(LrSchedule):
    """
    Warmup-Stable-Decay Schedule.
    Warms up, holds constant, then decays linearly over the last `decay_ratio` of steps.
    """
    decay_ratio: float = 0.8  # Fraction of training spent decaying (e.g., last 20%)

    def build(self, ctx: LrScheduleContext):
        remaining_steps = ctx.decay_steps
        decay_steps = int(remaining_steps * self.decay_ratio)
        stable_steps = remaining_steps - decay_steps

        # 2. Stable Phase (Peak Constant)
        stable_schedule = optax.constant_schedule(ctx.learning_rate)

        # 3. Decay Phase (Peak -> Min)
        decay_schedule = optax.linear_schedule(
                init_value=ctx.learning_rate,
                end_value=ctx.min_lr,
                transition_steps=decay_steps
                )
        return optax.join_schedules(
            [stable_schedule, decay_schedule],
            [stable_steps]
        )


def _build_manual_schedule(config, num_train_steps):
    if config.lr_schedule == "wsd":
        decay_ratio = getattr(config, 'decay_ratio', 0.8)
        ctx = LrScheduleContext(
                warmup_steps=int(config.warmup * num_train_steps) if isinstance(config.warmup, float) else (config.warmup or 0),
                decay_steps=num_train_steps,
                learning_rate=config.learning_rate,
                min_lr_ratio=config.min_lr_ratio or 0.0,
                min_lr=(config.min_lr_ratio or 0.0) * config.learning_rate
                )
        return WSDLrSchedule(decay_ratio=decay_ratio).build(ctx)
    else:
        return config.lr_scheduler(num_train_steps)


@OptimizerConfig.register_subclass("low_rank_orthogonal")
@dataclass(frozen=True)
class LROOConfig(OptimizerConfig):
    """
    Configuration for the Low Rank Orthogonal Optimizer.
    """
    learning_rate: float = 3e-4
    momentum: float = 0.95
    beta1: float = 0.8
    beta2: float = 0.98
    epsilon: float = 1e-8
    weight_decay: float = 0.0

    max_grad_norm: float | None = 1.0

    # lrou Specifics
    krylov_iter: int = 1
    rank_type: str = "constant"
    rank_val: int = 32

    # Design Strategy Flags
    embedding_strategy: str = "adam"
    lm_head_strategy: str = "adam"

    adam_learning_rate: Optional[float] = None

    adam_weight_decay: float = 0.1

    seed: int = 42

    def build(self, num_train_steps: int):
        print(f"Building optimizer: {self.__class__.__name__}")

        lr_schedule = _build_manual_schedule(self, num_train_steps)

        # Handle separate Adam LR
        if self.adam_learning_rate is not None:
            # Create a temp config just to get the schedule function
            adam_conf = dataclasses.replace(
                    self,
                    learning_rate=self.adam_learning_rate,
                    lr_schedule="cosine",
                    min_lr_ratio=0.1,
                    )
            adam_schedule = adam_conf.lr_scheduler(num_train_steps)
        else:
            adam_schedule = lr_schedule

        key = jax.random.key(self.seed)

        def optimizer_factory(learning_rate, adam_learning_rate):
            base_opt = low_rank_orthogonal_update(
                    lr=learning_rate,
                    key=key,
                    momentum=self.momentum,
                    beta1=self.beta1,
                    beta2=self.beta2,
                    krylov_iter=self.krylov_iter,
                    rank_type=self.rank_type,
                    rank_val=self.rank_val,
                    embedding_strategy=self.embedding_strategy,
                    lm_head_strategy=self.lm_head_strategy,
                    adam_lr=adam_learning_rate,
                    eps=self.epsilon,
                    weight_decay=self.weight_decay,
                    adam_weight_decay=self.adam_weight_decay,
                    mask=self.build_weight_decay_mask(),
                    )

            if self.max_grad_norm is not None:
                return optax.chain(
                        optax.clip_by_global_norm(self.max_grad_norm),
                        base_opt,
                        )
            else:
                return base_opt

        return optax.inject_hyperparams(optimizer_factory)(learning_rate=lr_schedule, adam_learning_rate=adam_schedule)


@OptimizerConfig.register_subclass("custom_muon")
@dataclass(frozen=True)
class WSDMuonConfig(OptimizerConfig):
    """
    Custom Muon Config that supports WSD Scheduling and separate Adam scheduling.
    """
    lr: float = 0.02
    adam_lr: float = 6e-4
    momentum: float = 0.95
    nesterov: bool = True
    backend_steps: int = 5
    weight_decay: float = 0.0
    adam_weight_decay: Optional[float] = None
    beta1: float = 0.8   # Adam Beta1
    beta2: float = 0.99  # Adam Beta2
    epsilon: float = 1e-8
    muon_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    use_kimi_scaling: bool = False

    # Added field for WSD schedule control
    decay_ratio: float = 0.8

    def build(self, num_train_steps):
        # 1. Muon Schedule (WSD/Linear)
        muon_schedule = _build_manual_schedule(self, num_train_steps)

        # 2. Adam Schedule (Force Cosine)
        adam_conf = dataclasses.replace(
            self,
            learning_rate=self.adam_lr,
            lr_schedule="cosine",
            min_lr_ratio=0.1
        )
        adam_schedule = adam_conf.lr_scheduler(num_train_steps)

        def optimizer(learning_rate, adam_lr_val):
            def muon_transform():
                components = []
                components.append(
                    scale_with_muon(
                        self.momentum, self.nesterov, self.backend_steps, self.muon_epsilon, self.use_kimi_scaling
                    )
                )
                if self.weight_decay > 0:
                    components.append(optax.add_decayed_weights(self.weight_decay, self.build_weight_decay_mask()))
                components.append(optax.scale(-learning_rate))
                return optax.chain(*components)

            def adamw_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon))

                adam_wd = self.adam_weight_decay if self.adam_weight_decay is not None else self.weight_decay
                if adam_wd > 0:
                    components.append(optax.add_decayed_weights(adam_wd, self.build_weight_decay_mask()))
                components.append(optax.scale(-adam_lr_val))
                return optax.chain(*components)

            transformations = {
                "muon": muon_transform(),
                "adamw": adamw_transform(),
            }

            return optax.multi_transform(
                transformations, partial(self.create_mask, use_kimi_scaling=self.use_kimi_scaling)
            )

        return optax.inject_hyperparams(optimizer)(
            learning_rate=muon_schedule,
            adam_lr_val=adam_schedule
        )

    def create_mask(self, params, use_kimi_scaling=True):
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            if "Embedding" in path_str or "lm_head" in path_str:
                return "adamw"
            elif isinstance(param, Linear):
                return dataclasses.replace(param, weight="muon", bias="adamw" if param.bias is not None else None)
            else:
                return "adamw"
        return jax.tree_util.tree_map(mask_fn, params, paths, is_leaf=lambda x: isinstance(x, Linear))



@dataclass(frozen=True)
class LraTrainConfig:
    """
    A Custom Training Config for LRA.
    Matches Marin's SimpleTrainConfig structure but adds axis_resources.
    """
    resources: Union[GpuConfig, TpuPodConfig]

    train_batch_size: Union[int, IntSchedule]
    num_train_steps: int
    learning_rate: float

    # --- Optional Fields matching SimpleTrainConfig ---
    data_seed: Optional[int] = None
    weight_decay: Optional[float] = None
    beta1: Optional[float] = None
    beta2: Optional[float] = None
    epsilon: Optional[float] = None
    max_grad_norm: Optional[float] = None

    warmup: Optional[float] = None
    decay: Optional[float] = None
    rewarmup: Optional[float] = None
    lr_schedule: Optional[str] = None
    min_lr_ratio: Optional[float] = None
    cycle_length: Union[int, List[int], None] = None

    z_loss_weight: Optional[float] = None
    ema_beta: Optional[float] = None
    skip_bad_steps: bool = False

    # Evaluation & Export
    steps_per_eval: Optional[int] = None
    steps_per_export: int = 10000
    steps_per_task_eval: Optional[int] = None
    steps_per_hf_export: Optional[int] = None

    per_device_eval_parallelism: Optional[int] = None
    max_eval_batches: Optional[int] = None

    # Checkpointing
    initialize_from_checkpoint_path: Optional[str] = None
    initialize_from_hf: Optional[str] = None
    reset_data_loader_on_init: bool = True
    allow_partial_checkpoint: bool = False

    # Advanced
    int8: bool = False

    optimizer_config: Optional[OptimizerConfig] = None

    # FIX: Use default_factory to ensure a valid WatchConfig object exists
    watch: WatchConfig = field(default_factory=WatchConfig)

    # Profiler
    profiler: bool = False
    profiler_start_step: int = 5
    profiler_num_steps: int = 100

    # --- CUSTOM FIELD ---
    # Marin's default_train might ignore this if it doesn't explicitly look for it,
    # but having it here prevents AttributeErrors if you patch defaults.py.
    axis_resources: Optional[Dict[str, Any]] = None
