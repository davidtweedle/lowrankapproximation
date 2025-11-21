from dataclasses import dataclass

import jax
import optax

from levanter.optim import OptimizerConfig

from marin.execution.executor import executor_main
from marin.resources import GpuConfig
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun

from experiments.llama import llama_75m

from lra_opt import low_rank_orthogonal_update, create_param_labels, LraTrainConfig


@dataclass(frozen=True)
class LowRankOrthogonalConfig(OptimizerConfig):
    beta1: float = 0.9
    beta2: float = 0.99
    eps: float = 1e-8
    eps_root: float = 0.0

    max_grad_norm: float | None = 1.0
    krylov_iter: int = 2
    rank_type: str = "sqrt"
    rank_val: int | None = None

    seed: int = 13

    def build(self, num_train_steps: int):
        print(f"Building optimizer: {self.__class__.__name__}")
        try:
            OptimizerConfig.register_subclass("low_rank_orthogonal")(LowRankOrthogonalConfig)
        except ValueError:
            pass
        param_label_fn = create_param_labels()

        def _optimizer(learning_rate):
            key = jax.random.key(self.seed)
            base_opt = low_rank_orthogonal_update(
                    lr=learning_rate,
                    key=key,
                    beta1=self.beta1,
                    beta2=self.beta2,
                    krylov_iter=self.krylov_iter,
                    rank_type=self.rank_type,
                    rank_val=self.rank_val,
                    param_label_fn=param_label_fn,
                    eps=self.eps,
                    eps_root=self.eps_root,
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
        return optax.inject_hyperparams(_optimizer)(
                learning_rate=self.lr_scheduler(num_train_steps)
                )

speedrun_config = SpeedrunConfig(
        author=Author(
            name="David Tweedle",
            affiliation="None",
            url="https://github.com/davidtweedle"
            ),
        description="75M parameter model with Low-rank orthogonal optimizer",
        model_config=llama_75m,
        train_config=LraTrainConfig(
            resources=GpuConfig(
                gpu_count=4,
                accelerator_type="A100",
                ),
            train_batch_size=128,
            num_train_steps=6000,
            steps_per_eval=2000,
            learning_rate=3e-4,
            weight_decay=0.1,
            axis_resources={
                'batch': 'data',
                'vocab': None,
                'mlp': None,
                'embed': None,
                'heads': None,
                'kv_heads': None,
                },
            optimizer_config=LowRankOrthogonalConfig(
                learning_rate=3e-4,
                weight_decay=0.1,
                lr_schedule='cosine',
                warmup=500,
                min_lr_ratio=0.1,
                ),
            ),
        )

speedrun_config.print_run_info()

if __name__ == "__main__":
    executor_main(steps=default_speedrun("llama_75m_low_rank_orthogonal", speedrun_config))
