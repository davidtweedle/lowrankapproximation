import os

from levanter.trainer import TrainerConfig
from levanter.optim import AdamConfig

# Model Config
from experiments.llama import llama_150m

from marin.execution.executor import executor_main
from marin.speedrun.speedrun import SpeedrunConfig, Author, default_speedrun
from marin.resources import GpuConfig
from lra_opt import LraTrainConfig

# --- Monkey Patch for Data Parallelism ---
# Ensures Adam uses the same replication strategy as your LRA run
_orig_init = TrainerConfig.__init__


def _new_init(self, *args, **kwargs):
    kwargs['axis_resources'] = {
            'batch': 'data',
            'vocab': None, 'mlp': None, 'embed': None, 'heads': None, 'kv_heads': None,
            }
    _orig_init(self, *args, **kwargs)


TrainerConfig.__init__ = _new_init

if __name__ == "__main__":
    # --- TUNING PARAMETERS ---
    # Default to 6e-4 (Standard for 150M), but allow sweep override
    LR = float(os.environ.get("TUNE_ADAM_LR", "6e-4"))
    WD = float(os.environ.get("TUNE_ADAM_WD", "0.1"))

    RUN_ID_SUFFIX = os.environ.get("RUN_ID_SUFFIX", "default")
    STEPS = int(os.environ.get("TUNE_STEPS", "4000"))
    BATCH_SIZE = 128

    print(f"--- LAUNCHING ADAM SWEEP: LR={LR}, WD={WD} ---")

    # Configure AdamW
    # Levanter's AdamConfig defaults to AdamW behavior (decoupled decay)
    opt_config = AdamConfig(
        learning_rate=LR,
        weight_decay=WD,
        beta1=0.9,
        beta2=0.95,
        epsilon=1e-8,
        warmup=0.1,
        lr_schedule='cosine',
        min_lr_ratio=0.1
    )

    train_config = LraTrainConfig(
        resources=GpuConfig(gpu_count=4, accelerator_type="A100"),
        train_batch_size=BATCH_SIZE,
        num_train_steps=STEPS,

        # Pass these for LraTrainConfig requirements
        learning_rate=LR,
        weight_decay=WD,

        # Skip heavy eval for sweep speed, but keep export for safety
        steps_per_eval=500,
        steps_per_export=10000,

        optimizer_config=opt_config,

        axis_resources={
            'batch': 'data',
            'vocab': None, 'mlp': None, 'embed': None, 'heads': None, 'kv_heads': None,
        },
    )

    # Setup WandB
    os.environ["WANDB_PROJECT"] = "lroo"
    os.environ["WANDB_ENTITY"] = "david-tweedle-none"

    speedrun_conf = SpeedrunConfig(
        author=Author(name="David Tweedle", affiliation="Indep", url="https://github.com/davidtweedle"),
        description=f"Baseline AdamW: LR={LR} WD={WD}",
        model_config=llama_150m,
        train_config=train_config
    )

    executor_main(steps=default_speedrun(f"adam_{RUN_ID_SUFFIX}", speedrun_conf))
