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

# Use 125M config (Ensure this exists in your experiments/llama.py, otherwise use llama_nano)
try:
    from experiments.llama import llama_150m
except ImportError:
    from experiments.llama import llama_nano as llama_150m
    print("WARNING: llama_150m not found, falling back to llama_nano")

from lra_opt import low_rank_orthogonal_update, create_param_labels, LraTrainConfig

# --- Monkey Patch for Axis Resources (Legacy Safety) ---
_orig_init = TrainerConfig.__init__
def _new_init(self, *args, **kwargs):
    if 'axis_resources' not in kwargs or kwargs['axis_resources'] is None:
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
        
        # Handle separate Adam LR
        if self.adam_learning_rate is not None:
            # Create a temp config just to get the schedule function
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

def run_variant(variant_id):
    print(f"--- LAUNCHING SPRINT: VARIANT {variant_id} ---")
    
    # Defaults
    base_lr = 0.02 
    adam_lr = 6e-4
    wd = 0.1
    
    # --- FIX: Correct String Constants ---
    STRAT_ADAM = 'adam'
    STRAT_QR = 'qr_with_pivot' # Must match linalg.py
    
    if variant_id == 'A':
        desc = "Baseline: Sketch=32, Fused, Embed=Adam, Head=Adam"
        embed_strat = STRAT_ADAM
        head_strat = STRAT_ADAM
        sketch = 32
        
    elif variant_id == 'B':
        desc = "HighRank: Sketch=64, Fused, Embed=Adam, Head=Adam"
        embed_strat = STRAT_ADAM
        head_strat = STRAT_ADAM
        sketch = 64
        
    elif variant_id == 'C':
        desc = "EmbedQR: Sketch=32, Fused, Embed=PivotedQR, Head=Adam"
        embed_strat = STRAT_QR
        head_strat = STRAT_ADAM
        sketch = 32

    elif variant_id == 'D':
        desc = "FullSpecial: Sketch=32, Fused, Embed=PivotedQR, Head=PivotedQR"
        embed_strat = STRAT_QR
        head_strat = STRAT_QR
        sketch = 32
        
    else:
        print(f"Error: Unknown variant '{variant_id}'. Choose A, B, C, or D.")
        return

    # Build Optimizer Config
    opt_config = LowRankOrthogonalConfig(
        learning_rate=base_lr,
        adam_learning_rate=adam_lr,
        weight_decay=wd,
        lr_schedule='cosine',
        warmup=0.1, 
        min_lr_ratio=0.1,
        rank_val=sketch,
        embedding_strategy=embed_strat,
        lm_head_strategy=head_strat,
        krylov_iter=1
    )

    # Build Training Config
    train_config = LraTrainConfig(
        resources=GpuConfig(gpu_count=4, accelerator_type="A100"),
        train_batch_size=256,
        num_train_steps=2000, 
        
        # --- FIX: Add Missing Fields ---
        learning_rate=base_lr, 
        weight_decay=wd,
        
        # Disable Eval for Sprint (Set > num_train_steps)
        steps_per_eval=2001, 
        # -------------------------------

        optimizer_config=opt_config,
        
        # Axis resources (Redundant due to patch, but safe to keep)
        axis_resources={ 
            'batch': 'data',
            'vocab': None,
            'mlp': None,
            'embed': None,
            'heads': None,
            'kv_heads': None,
        },
    )

    # Setup WandB Environment
    os.environ["WANDB_PROJECT"] = "lra-optimizer"
    os.environ["WANDB_RUN_GROUP"] = "lra_sprint_150m"
    os.environ["WANDB_TAGS"] = f"sprint,variant_{variant_id}"

    speedrun_conf = SpeedrunConfig(
        author=Author(name="David Tweedle", affiliation="Indep", url="https://github.com/davidtweedle"),
        description=desc,
        model_config=llama_150m,
        train_config=train_config
    )

    # Force Run ID to be unique per variant
    executor_main(steps=default_speedrun(f"sprint_150m_{variant_id}", speedrun_conf))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        variant_id = sys.argv[1]
        # Clean up argv so Levanter doesn't get confused
        del sys.argv[1]
        run_variant(variant_id)
    else:
        print("Please provide a variant ID (A, B, C, D)")
