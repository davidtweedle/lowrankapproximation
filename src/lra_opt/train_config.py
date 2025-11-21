import dataclasses
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union, List

from levanter.callbacks.watch import WatchConfig
from levanter.optim import OptimizerConfig
from levanter.schedule import IntSchedule

from marin.resources import GpuConfig, TpuPodConfig

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
    
    # --- CUSTOM FIELD FOR LRA ---
    # Marin's default_train might ignore this if it doesn't explicitly look for it,
    # but having it here prevents AttributeErrors if you patch defaults.py.
    axis_resources: Optional[Dict[str, Any]] = None
