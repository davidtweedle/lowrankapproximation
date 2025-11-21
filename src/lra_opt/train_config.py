from dataclasses import dataclass
from typing import Optional, Dict, Any, Union, List

from levanter.trainer import TrainerConfig
from levanter.optim import OptimizerConfig
from levanter.tracker.wandb import WandbConfig
from marin.resources import GpuConfig, TpuPodConfig


def _get_default_watch_config():
    try:
        for f in fields(TrainerConfig):
            if f.name == 'watch':
                return f.type()
    except Exception as e:
        print(f"WARNING: Could not reflectively create watch config: {e}")

    @dataclass
    class DummyWatch:
        is_enabled: bool = False
    return DummyWatch()

@dataclass
class LraTrainConfig:
    """
    A Custom Training Config for LRA that supports axis_resources (Sharding Control).
    Based on Marin's SimpleTrainConfig.
    """
    resources: Union[GpuConfig, TpuPodConfig]
    train_batch_size: int
    num_train_steps: int
    learning_rate: float
    weight_decay: float
    steps_per_eval: int
    data_seed: int = 0
    per_device_parallelism: int = -1
    per_device_eval_parallelism: int = -1
    steps_per_export: Optional[int] = None
    steps_per_task_eval: Optional[int] = None
    steps_per_hf_export: Optional[int] = None
    hf_save_path: Optional[str] = None
    hf_upload: Optional[str] = None

    max_eval_batches: Optional[int] = None

    int8: bool = False

    wandb: Optional[Dict[str, Any]] = None
    watch: Optional[List[str]] = None
    profiler: Optional[str] = None
    profiler_start_step: int = -1
    profiler_num_steps: int = -1

    z_loss_weight: float = 0.0
    loss_scale: Optional[float] = None

    ema_beta: Optional[float] = None

    reset_data_loader_on_init: bool = True

    load_checkpoint_path: Optional[str] = None
    allow_partial_checkpoint: Optional[str] = None
    initialize_from_hf: Optional[str] = None
    initialize_from_checkpoint_path: Optional[str] = None

    per_device_eval_parallelism: Optional[bool] = None
    optimizer_config: Optional[OptimizerConfig] = None
    # --- ADDED FEATURE: Sharding Control ---
    # Allows forcing Data Parallelism by setting model axes to None
    axis_resources: Optional[Dict[str, Any]] = None
    # ---------------------------------------

    def to_trainer_config(self) -> TrainerConfig:
        """Converts this simple config into a full Levanter TrainerConfig."""
        # Levanter calculates per_device sizes from global, but we provide a fallback
        # Note: We rely on Levanter's internal defaults for most things.
        wandb_config = None
        if self.wandb is not None:
            wandb_config = WandbConfig(**self.wandb)
        watch_config = _get_default_watch_config()
        return TrainerConfig(
            num_train_steps=self.num_train_steps,
            train_batch_size=self.train_batch_size,
            # Pass the optimizer config (which contains the scheduler logic)
            optimizer=self.optimizer_config,
            # Evaluation settings
            steps_per_eval=self.steps_per_eval,
            steps_per_save=self.steps_per_export,
            per_device_parallelism=self.per_device_parallelism,
            per_device_eval_parallelism=self.per_device_eval_parallelism,
            ema_decay=self.ema_beta,
            load_checkpoint_path=self.load_checkpoint_path,
            wandb=wandb_config,
            watch=watch_config,
            # --- PASS THE SHARDING CONFIG ---
            axis_resources=self.axis_resources,
        )
