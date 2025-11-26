import os
import subprocess
import itertools

# --- WINNING HYPERPARAMS ---
BEST_LRA_LR = "0.02"
BEST_ADAM_LR = "0.001"

# --- EXPLORATION GRID ---
# 64 is added to check if 32 was bottling-necking performance
RANKS = [8, 16, 32, 64]
# 0 = Fast/Noisy, 1 = Slower/Accurate
KRYLOV_ITERS = [0, 1]


def run_sweep():
    # Setup Environment
    env = os.environ.copy()
    env["WANDB_MODE"] = "online"
    env["WANDB_PROJECT"] = "lroo"
    env["WANDB_ENTITY"] = "david-tweedle-none"  # Your correct entity

    # Filter: Only run training steps (saves time if you don't need eval)
    # Or keep standard if you want validation curves

    combinations = list(itertools.product(RANKS, KRYLOV_ITERS))

    for i, (rank, k) in enumerate(combinations):
        run_id = f"rank{rank}_krylov{k}_bestlr"
        print(f"\n>>> [{i+1}/{len(combinations)}] Launching: {run_id}")

        # Pass Params
        env["RUN_ID_SUFFIX"] = run_id
        env["TUNE_LRA_LR"] = BEST_LRA_LR
        env["TUNE_ADAM_LR"] = BEST_ADAM_LR

        # The Variables
        env["TUNE_RANK"] = str(rank)
        env["TUNE_KRYLOV"] = str(k)

        # Check if done (Skip existing logs)
        log_file = f"sweep_logs/{run_id}.log"
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                if "Executor run took" in f.read():
                    print("    Skipping (Completed)")
                    continue

        # Cleanup Previous Failures
        subprocess.run(f"rm -rf /workspace/marin_logs/checkpoints/speedrun/tune_{run_id}", shell=True)

        try:
            os.makedirs("sweep_logs", exist_ok=True)
            with open(log_file, "w") as f:
                subprocess.run(
                    ["python", "experiments/speedrun/lra_svd_run/run.py"],
                    env=env,
                    check=True,
                    stdout=f,
                    stderr=subprocess.STDOUT
                )
        except subprocess.CalledProcessError:
            print(f"    !!! FAILED !!! Check log: {log_file}")
            subprocess.run(["ray", "stop", "--force"])


if __name__ == "__main__":
    run_sweep()
