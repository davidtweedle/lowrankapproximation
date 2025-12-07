import os
import subprocess
import itertools

# --- HYPERPARAMETER GRID ---
# Matrix Learning Rates (for Muon and low-rank Body)
# Paper suggests ~0.016. We bracket that.
matrix_lrs = [0.035]

# Adam Learning Rates (for Embeddings/Heads/etc)
adam_lr_ratios = [0.4]

# --- CONSTANTS ---
sketch_vals = [2, 4, 8, 16]
rank_vals = [32, 64]
krylovs = [0, 2]


def run_sweep():
    # Global Cleanup before starting
    print(">>> Killing old Ray processes...")
    subprocess.run(["ray", "stop", "--force"])

    combinations = list(itertools.product(matrix_lrs, adam_lr_ratios, sketch_vals, rank_vals, krylovs))
    total = len(combinations)

    count = 0
    for i, (mlr, alr, sketch_val, rank_val, krylov) in enumerate(combinations):

        # ==========================================
        # 2. RUN LRA (Linear Matrix / Cosine Adam)
        # ==========================================
        count += 1
        run_id = f"lroo_M{mlr}_A{alr}_S{sketch_val}_R{rank_val}_K{krylov}_test_sweep"
        print(f"\n=== [{count}/{total}] STARTING LRA: {run_id} ===")

        env = os.environ.copy()
        env["TUNE_LRA_LR"] = str(mlr)
        env["TUNE_ADAM_LR_RATIO"] = str(alr)
        env["TUNE_SKETCH"] = str(sketch_val)
        env["TUNE_RANK"] = str(rank_val)
        env["TUNE_KRYLOV"] = str(krylov)
        env["TUNE_STEPS"] = "4000"
        env["RUN_ID_SUFFIX"] = run_id

        # Force Online Logging
        env["WANDB_MODE"] = "online"
        env["WANDB_ENTITY"] = "david-tweedle-none"
        env["WANDB_PROJECT"] = "lroo"

        # Clean Checkpoints
        subprocess.run(f"rm -rf /workspace/marin_logs/checkpoints/speedrun/lra_{run_id}", shell=True)

        try:
            # Run LRA Script
            subprocess.run(
                ["python", "experiments/speedrun/lra_svd_run/run.py"],
                env=env,
                check=True,
            )
            print("  > LRA Finished Successfully.")

        except subprocess.CalledProcessError:
            print("  > LRA FAILED. Check logs.")
            subprocess.run(["ray", "stop", "--force"])


if __name__ == "__main__":
    run_sweep()
