import sys
import os
import jax
import optax
from dataclasses import dataclass
import dataclasses
from typing import Any, Optional

from levanter.optim import OptimizerConfig
from levanter.trainer import TrainerConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import executor_main
from marin.speedrun.speedrun import SpeedrunConfig, Author, default_speedrun
from marin.resources import GpuConfig

# Use 150M config
from experiments.llama import llama_150m

from lra_opt import low_rank_orthogonal_update, create_param_labels, LraTrainConfig

# --- Monkey Patch for Axis Resources (Legacy Safety) ---
_orig_init = TrainerConfig.__init__


def _new_init(self, *args, **kwargs):
    kwargs['axis_resources'] = {
            'batch': 'data',
            'vocab': None,
            'mlp': None,
            'embed': None,
            'heads': None,
            'kv_heads': None,
            }
    _orig_init(self, *args, **kwargs)


TrainerConfig.__init__ = _new_init


@OptimizerConfig.register_subclass("low_rank_orthogonal")
@dataclass(frozen=True)
class LowRankOrthogonalConfig(OptimizerConfig):
    """
    Configuration for the Low Rank Orthogonal Optimizer.
    """
    learning_rate: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.99
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

        lr_schedule = self.lr_scheduler(num_train_steps)

        # Handle separate Adam LR
        if self.adam_learning_rate is not None:
            # Create a temp config just to get the schedule function
            adam_conf = dataclasses.replace(self, learning_rate=self.adam_learning_rate)
            adam_schedule = adam_conf.lr_scheduler(num_train_steps)
        else:
            adam_schedule = lr_schedule

        key = jax.random.key(self.seed)

        def optimizer_factory(learning_rate):
            base_opt = low_rank_orthogonal_update(
                    lr=learning_rate,
                    key=key,
                    beta1=self.beta1,
                    beta2=self.beta2,
                    krylov_iter=self.krylov_iter,
                    rank_type=self.rank_type,
                    rank_val=self.rank_val,
                    embedding_strategy=self.embedding_strategy,
                    lm_head_strategy=self.lm_head_strategy,
                    adam_lr=adam_schedule,
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

        return optax.inject_hyperparams(optimizer_factory)(learning_rate=lr_schedule)


def run_variant(variant_id):    # --- TUNING ---
    LRA_LR = float(os.environ.get("TUNE_LRA_LR", "0.02"))
    ADAM_LR = float(os.environ.get("TUNE_ADAM_LR", "0.003"))

    # Fixed for this campaign
    KRYLOV = 0
    RANK = 32
    SCHEDULE = "linear"

    RUN_ID_SUFFIX = os.environ.get("RUN_ID_SUFFIX", "default")
    STEPS = int(os.environ.get("TUNE_STEPS", "4000"))

    print(f">>> LRA RUN: LR={LRA_LR}, Adam={ADAM_LR}, K={KRYLOV}, R={RANK}, S={SCHEDULE}")

    opt_config = LowRankOrthogonalConfig(
        learning_rate=LRA_LR,
        adam_learning_rate=ADAM_LR,

        weight_decay=0.0,      # Zero WD for Matrix updates (Like Muon)
        adam_weight_decay=0.1, # (Passed to logic, see note above)

        lr_schedule=SCHEDULE,  # Linear
        warmup=0.1,
        min_lr_ratio=0.0,      # Linear decay to 0

        rank_val=RANK,
        krylov_iter=KRYLOV,
        embedding_strategy="adam",
        lm_head_strategy="adam"
    )

    train_config = LraTrainConfig(
        resources=GpuConfig(gpu_count=4, accelerator_type="A100"),
        train_batch_size=128,
        num_train_steps=STEPS,
        learning_rate=LRA_LR,
        weight_decay=0.1,  # General flag

        steps_per_eval=1000,
        steps_per_export=5000,

        optimizer_config=opt_config,
        axis_resources={
            'batch': 'data',
            'vocab': None, 'mlp': None, 'embed': None, 'heads': None, 'kv_heads': None,
        },
    )

    os.environ["WANDB_PROJECT"] = "lra-optimizer"
    os.environ["WANDB_RUN_GROUP"] = "combined_sweep"
    os.environ["WANDB_TAGS"] = f"lra,rank{RANK},k{KRYLOV},linear"
    os.environ["WANDB_ENTITY"] = "david-tweedle-none"

    speedrun_conf = SpeedrunConfig(
        author=Author(name="David Tweedle", affiliation="Indep", url="https://github.com/davidtweedle"),
        description=f"LRA Sweep: L={LRA_LR}, A={ADAM_LR}",
        model_config=llama_150m,
        train_config=train_config
    )

    executor_main(steps=default_speedrun(f"lra_{RUN_ID_SUFFIX}", speedrun_conf))


if __name__ == '__main__':
    variant_id = "tune"
    if len(sys.argv) > 1:
        variant_id = sys.argv[1]
    run_variant(variant_id)
