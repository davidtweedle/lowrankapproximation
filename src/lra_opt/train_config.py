from dataclasses import dataclass
from typing import Optional, Dict, Any, Union

from levanter.trainer import TrainerConfig
from levanter.optim import OptimizerConfig
from marin.resources import GpuConfig, TpuPodConfig

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
    steps_per_export: Optional[int] = None
    steps_per_hf_export: Optional[int] = None
    optimizer_config: Optional[OptimizerConfig] = None
    # --- ADDED FEATURE: Sharding Control ---
    # Allows forcing Data Parallelism by setting model axes to None
    axis_resources: Optional[Dict[str, Any]] = None
    # ---------------------------------------

    def to_trainer_config(self) -> TrainerConfig:
        """Converts this simple config into a full Levanter TrainerConfig."""
        # Levanter calculates per_device sizes from global, but we provide a fallback
        # Note: We rely on Levanter's internal defaults for most things.
        return TrainerConfig(
            num_train_steps=self.num_train_steps,
            train_batch_size=self.train_batch_size,
            # Pass the optimizer config (which contains the scheduler logic)
            optimizer=self.optimizer_config,
            # Evaluation settings
            steps_per_eval=self.steps_per_eval,
            steps_per_save=self.steps_per_export,
            # --- PASS THE SHARDING CONFIG ---
            axis_resources=self.axis_resources,
        )
