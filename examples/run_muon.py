import os

from levanter.trainer import TrainerConfig
from lra_opt import WSDMuonConfig, LraTrainConfig

# Model Config
from experiments.llama import llama_150m

from marin.execution.executor import executor_main
from marin.speedrun.speedrun import SpeedrunConfig, Author, default_speedrun
from marin.resources import GpuConfig

# --- Monkey Patch for Data Parallelism ---
_orig_init = TrainerConfig.__init__


def _new_init(self, *args, **kwargs):
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
    ADAM_LR_RATIO = float(os.environ.get("TUNE_ADAM_LR_RATIO", "0.2"))
    RUN_ID_SUFFIX = os.environ.get("RUN_ID_SUFFIX", "default")
    STEPS = int(os.environ.get("TUNE_STEPS", "4000"))

    print(f"--- LAUNCHING MUON SWEEP: MuonLR={MUON_LR}, AdamLR={MUON_LR * ADAM_LR_RATIO} ---")

    # Configure Muon
    opt_config = WSDMuonConfig(
        lr=MUON_LR,             # The matrix learning rate
        adam_lr_ratio=ADAM_LR_RATIO,        # The embedding/bias learning rate
        lr_schedule="wsd",
        decay_ratio=0.8,
        warmup=0,
        min_lr_ratio=0.0,
        momentum=0.95,
        nesterov=True,
        beta1=0.8,
        backend_steps=5,
        weight_decay=0.0,       # Muon paper suggests 0 WD for matrices
        adam_weight_decay=0.1,  # But standard WD for embeddings
    )

    train_config = LraTrainConfig(
        resources=GpuConfig(gpu_count=4, accelerator_type="A100"),
        train_batch_size=128,  # Match your LRA sweep
        num_train_steps=STEPS,

        # Levanter requires these fields, though MuonConfig handles the actual scheduling
        learning_rate=MUON_LR,
        weight_decay=0.0,

        # Optimization for Sweep: Skip heavy evaluation
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
        description=f"Muon: {MUON_LR}",
        model_config=llama_150m,
        train_config=train_config
    )

    executor_main(steps=default_speedrun(f"muon_{RUN_ID_SUFFIX}", speedrun_conf))
