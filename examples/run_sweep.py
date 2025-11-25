import sys
import os
import jax
import optax
from dataclasses import dataclass
import dataclasses
from typing import Any, Optional

from levanter.optim import OptimizerConfig
from levanter.trainer import TrainerConfig
from marin.execution.executor import executor_main
from marin.speedrun.speedrun import SpeedrunConfig, Author, default_speedrun
from marin.resources import GpuConfig

# Use 125M/150M config
from experiments.llama import llama_150m

from lra_opt import low_rank_orthogonal_update, create_param_labels, LraTrainConfig

# --- Monkey Patch for Axis Resources ---
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
    learning_rate: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.99
    epsilon: float = 1e-8
    weight_decay: float = 0.0
    max_grad_norm: float | None = 1.0
    
    # LRA Specifics
    krylov_iter: int = 1
    rank_type: str = "constant" 
    rank_val: int = 32 
    
    # Design Strategy Flags
    embedding_strategy: str = "adam"
    lm_head_strategy: str = "adam"

    adam_learning_rate: Optional[float] = None
    seed: int = 42

    def build(self, num_train_steps: int):
        print(f"Building optimizer: {self.__class__.__name__}")
        
        lr_schedule = self.lr_scheduler(num_train_steps)
        
        if self.adam_learning_rate is not None:
            adam_conf = dataclasses.replace(self, learning_rate=self.adam_learning_rate)
            adam_schedule = adam_conf.lr_scheduler(num_train_steps)
        else:
            adam_schedule = lr_schedule
        
        param_label_fn = create_param_labels(self.embedding_strategy, self.lm_head_strategy)
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

if __name__ == "__main__":
    # --- TUNING PARAMETERS FROM ENV ---
    RUN_ID_SUFFIX = os.environ.get("RUN_ID_SUFFIX", "default")
    
    # LRA Settings
    LRA_LR = float(os.environ.get("TUNE_LRA_LR", "0.02"))
    KRYLOV = int(os.environ.get("TUNE_KRYLOV", "1"))
    
    # Adam Settings
    ADAM_LR = float(os.environ.get("TUNE_ADAM_LR", "6e-4"))
    
    # Fixed for this sweep
    STEPS = 4000 
    SKETCH_RANK = 32
    
    print(f">>> CONFIGURING RUN: LRA_LR={LRA_LR}, ADAM_LR={ADAM_LR}, KRYLOV={KRYLOV}")

    opt_config = LowRankOrthogonalConfig(
        learning_rate=LRA_LR,
        adam_learning_rate=ADAM_LR,
        weight_decay=0.1,
        lr_schedule='cosine',
        warmup=0.1, # 10% warmup
        min_lr_ratio=0.1,
        
        rank_val=SKETCH_RANK,
        krylov_iter=KRYLOV,
        
        # Fixed Strategies
        embedding_strategy="adam",
        lm_head_strategy="adam"
    )

    train_config = LraTrainConfig(
        resources=GpuConfig(gpu_count=4, accelerator_type="A100"),
        train_batch_size=128,
        num_train_steps=STEPS, 
        
        learning_rate=LRA_LR, 
        weight_decay=0.1,
        
        # Checkpoint less frequently to save disk space during sweep
        steps_per_export=2000,
        steps_per_eval=2001, # Disable eval for speed
        
        optimizer_config=opt_config,
        
        # Explicit Axis Resources
        axis_resources={ 
            'batch': 'data',
            'vocab': None, 'mlp': None, 'embed': None, 'heads': None, 'kv_heads': None,
        },
    )

    # Setup WandB
    os.environ["WANDB_PROJECT"] = "lra-optimizer"
    os.environ["WANDB_RUN_GROUP"] = "tuning_sweep_v1"
    os.environ["WANDB_TAGS"] = f"sweep,k{KRYLOV}"

    speedrun_conf = SpeedrunConfig(
        author=Author(name="David Tweedle", affiliation="Indep", url="https://github.com/davidtweedle"),
        description=f"Sweep: LRA={LRA_LR}, Adam={ADAM_LR}, K={KRYLOV}",
        model_config=llama_150m,
        train_config=train_config
    )

    executor_main(steps=default_speedrun(f"tune_{RUN_ID_SUFFIX}", speedrun_conf))
