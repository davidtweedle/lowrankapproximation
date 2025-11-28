import os
import jax
from dataclasses import dataclass
from typing import Optional

from levanter.trainer import TrainerConfig
from levanter.optim.muon import MuonConfig

# Model Config
from experiments.llama import llama_150m

from marin.execution.executor import executor_main
from marin.speedrun.speedrun import SpeedrunConfig, Author, default_speedrun
from marin.resources import GpuConfig
from lra_opt.train_config import LraTrainConfig

# --- Monkey Patch for Data Parallelism ---
_orig_init = TrainerConfig.__init__


def _new_init(self, *args, **kwargs):
    if 'axis_resources' not in kwargs or kwargs['axis_resources'] is None:
        kwargs['axis_resources'] = {
                'batch': 'data',
                'vocab': None, 'mlp': None, 'embed': None, 'heads': None, 'kv_heads': None,
                }
    _orig_init(self, *args, **kwargs)


TrainerConfig.__init__ = _new_init

if __name__ == "__main__":
    # --- TUNING PARAMETERS FROM ENV ---
    # Defaults based on Muon paper recommendations
    MUON_LR = float(os.environ.get("TUNE_MUON_LR", "0.02"))
    ADAM_LR = float(os.environ.get("TUNE_ADAM_LR", "0.0006"))
    RUN_ID_SUFFIX = os.environ.get("RUN_ID_SUFFIX", "default")
    STEPS = int(os.environ.get("TUNE_STEPS", "4000"))

    print(f"--- LAUNCHING MUON SWEEP: MuonLR={MUON_LR}, AdamLR={ADAM_LR} ---")

    # Configure Muon
    opt_config = MuonConfig(
        lr=MUON_LR,             # The matrix learning rate
        adam_lr=ADAM_LR,        # The embedding/bias learning rate
        momentum=0.95,
        nesterov=True,
        backend_steps=5,
        weight_decay=0.0,       # Muon paper suggests 0 WD for matrices
        adam_weight_decay=0.1,  # But standard WD for embeddings
        lr_schedule="linear",
        min_lr_ratio=0.0,
    )

    train_config = LraTrainConfig(
        resources=GpuConfig(gpu_count=4, accelerator_type="A100"),
        train_batch_size=128,  # Match your LRA sweep
        num_train_steps=STEPS,

        # Levanter requires these fields, though MuonConfig handles the actual scheduling
        learning_rate=MUON_LR,
        weight_decay=0.0,

        # Optimization for Sweep: Skip heavy evaluation
        steps_per_eval=1000,
        steps_per_export=5000,

        optimizer_config=opt_config,

        axis_resources={
            'batch': 'data',
            'vocab': None, 'mlp': None, 'embed': None, 'heads': None, 'kv_heads': None,
        },
    )

    # Setup WandB
    os.environ["WANDB_PROJECT"] = "lra-optimizer"
    os.environ["WANDB_RUN_GROUP"] = "muon_sweep"
    os.environ["WANDB_TAGS"] = "muon,tuning"
    os.environ["WANDB_ENTITY"] = "david-tweedle-none"

    speedrun_conf = SpeedrunConfig(
        author=Author(name="David Tweedle", affiliation="Indep", url="https://github.com/davidtweedle"),
        description=f"Muon Sweep: M={MUON_LR}, A={ADAM_LR}",
        model_config=llama_150m,
        train_config=train_config
    )

    executor_main(steps=default_speedrun(f"muon_{RUN_ID_SUFFIX}", speedrun_conf))
