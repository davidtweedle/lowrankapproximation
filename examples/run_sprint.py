import os

from levanter.trainer import TrainerConfig
from marin.execution.executor import executor_main
from marin.speedrun.speedrun import SpeedrunConfig, Author, default_speedrun
from marin.resources import GpuConfig

# Use 150M config
from experiments.llama import llama_150m

from lra_opt import LraTrainConfig, LROOConfig

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


if __name__ == "__main__":

    LRA_LR = float(os.environ.get("TUNE_LRA_LR", "0.02"))
    ADAM_LR = float(os.environ.get("TUNE_ADAM_LR", "0.003"))

    KRYLOV = int(os.environ.get("TUNE_KRYLOV", "0"))
    RANK = int(os.environ.get("TUNE_RANK", "32"))

    RUN_ID_SUFFIX = os.environ.get("RUN_ID_SUFFIX", "default")
    STEPS = int(os.environ.get("TUNE_STEPS", "4000"))


    print(f">>> LRA RUN: LR={LRA_LR}, Adam={ADAM_LR}, K={KRYLOV}, R={RANK}, S={SCHEDULE}")

    opt_config = LROOConfig(
        learning_rate=LRA_LR,
        adam_learning_rate=ADAM_LR,

        weight_decay=0.0,      # Zero WD for Matrix updates (Like Muon)
        adam_weight_decay=0.1, # (Passed to logic, see note above)

        lr_schedule="wsd",  # Linear
        decay_ratio=0.8,
        warmup=0,
        min_lr_ratio=0.0,      # Linear decay to 0
        momentum=0.95,
        beta1=0.8,

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
        weight_decay=0.0,  # General flag

        steps_per_eval=1000,
        steps_per_export=5000,

        optimizer_config=opt_config,
        axis_resources={
            'batch': 'data',
            'vocab': None, 'mlp': None, 'embed': None, 'heads': None, 'kv_heads': None,
        },
    )

    os.environ["WANDB_PROJECT"] = "lroo"
    os.environ["WANDB_ENTITY"] = "david-tweedle-none"

    speedrun_conf = SpeedrunConfig(
        author=Author(name="David Tweedle", affiliation="Indep", url="https://github.com/davidtweedle"),
        description=f"LROO LR: {LRA_LR}",
        model_config=llama_150m,
        train_config=train_config
    )

    executor_main(steps=default_speedrun(f"lra_{RUN_ID_SUFFIX}", speedrun_conf))
