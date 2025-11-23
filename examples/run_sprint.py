import sys
import jax
import optax
from dataclasses import dataclass
import dataclasses
from typing import Any, Optional

from levanter.optim import OptimizerConfig
from levanter.callbacks.watch import WatchConfig
from levanter.trainer import TrainerConfig

# Patch TrainerConfig for axis_resources if needed (Legacy support)
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

from marin.execution.executor import executor_main
from marin.speedrun.speedrun import SpeedrunConfig, Author
from marin.resources import GpuConfig


# Use 125M config for the sprint (Standard benchmark size)
from experiments.llama import llama_125m 

from lra_opt import low_rank_orthogonal_update, create_param_labels, LraTrainConfig

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
    embedding_strategy: str = "adam" # 'adam' or 'pivoted_qr'
    lm_head_strategy: str = "adam"   # 'adam' or 'pivoted_qr'

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
        
        # Pass strategies to label function
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
                    embedding_strategy=self.embedding_strategy, # Pass down
                    lm_head_strategy=self.lm_head_strategy,     # Pass down
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
                
        return optax.inject_hyperparams(_optimizer)(learning_rate=lr_schedule)

def run_variant(variant_id):
    print(f"--- LAUNCHING SPRINT: VARIANT {variant_id} ---")
    
    # Common Sprint Defaults
    base_lr = 0.02 # High LR for Spectral methods
    adam_lr = 6e-4
    wd = 0.1
    
    if variant_id == 'A':
        desc = "Baseline: Sketch=32, Fused, Embed=Adam, Head=Adam"
        embed_strat = 'adam'
        head_strat = 'adam'
        sketch = 32
        
    elif variant_id == 'B':
        desc = "HighRank: Sketch=64, Fused, Embed=Adam, Head=Adam"
        embed_strat = 'adam'
        head_strat = 'adam'
        sketch = 64
        
    elif variant_id == 'C':
        desc = "EmbedQR: Sketch=32, Fused, Embed=PivotedQR, Head=Adam"
        embed_strat = 'pivoted_qr'
        head_strat = 'adam'
        sketch = 32

    elif variant_id == 'D':
        desc = "FullSpecial: Sketch=32, Fused, Embed=PivotedQR, Head=PivotedQR"
        embed_strat = 'pivoted_qr'
        head_strat = 'pivoted_qr'
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
        num_train_steps=2000, # Sprint Length
        learning_rate=base_lr,
        optimizer_config=opt_config,
        axis_resources={ # Explicit axis resources for safety
            'batch': 'data',
            'vocab': None,
            'mlp': None,
            'embed': None,
            'heads': None,
            'kv_heads': None,
        },
        # Enable W&B
        watch=WatchConfig(
            group_name="lra_sprint_125m",
            project_name="lra-optimizer",
            tags=["sprint", f"variant_{variant_id}", embed_strat]
        )
    )

    speedrun_conf = SpeedrunConfig(
        author=Author(name="David Tweedle", affiliation="Indep", url="https://github.com/davidtweedle"),
        description=desc,
        model_config=llama_125m,
        train_config=train_config
    )

    executor_main(steps=speedrun.default_speedrun(f"sprint_{variant_id}", speedrun_conf))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python examples/speedrun_low_rank_optimizer.py [A|B|C|D]")
        sys.exit(1)
    run_variant(sys.argv[1])
