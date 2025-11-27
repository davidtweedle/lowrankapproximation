import os
import subprocess
import itertools
import time

# --- GRID ---
lrs = [3e-4, 6e-4, 1e-3, 3e-3]
wds = [0.1, 0.01]


def run_sweep():
    # Clear Ray state
    subprocess.run(["ray", "stop", "--force"])

    combinations = list(itertools.product(lrs, wds))
    print(f"Total Adam Runs: {len(combinations)}")

    for i, (lr, wd) in enumerate(combinations):
        run_id = f"LR{lr}_WD{wd}"
        print(f"\n[{i+1}/{len(combinations)}] Starting: {run_id}")

        # Pass params
        env = os.environ.copy()
        env["TUNE_ADAM_LR"] = str(lr)
        env["TUNE_ADAM_WD"] = str(wd)
        env["RUN_ID_SUFFIX"] = run_id

        # Force WandB settings
        env["WANDB_MODE"] = "online"
        env["WANDB_ENTITY"] = "david-tweedle-none"
        env["WANDB_PROJECT"] = "lroo"

        # Clean up previous run with same ID
        subprocess.run(f"rm -rf /workspace/marin_logs/checkpoints/speedrun/adam_{run_id}", shell=True)

        try:
            log_file = f"sweep_logs/adam_{run_id}.log"
            os.makedirs("sweep_logs", exist_ok=True)

            with open(log_file, "w") as f:
                subprocess.run(
                    ["python", "experiments/speedrun/lra_svd_run/run_adam.py"],
                    env=env,
                    check=True,
                    stdout=f,
                    stderr=subprocess.STDOUT
                )
            print(f"  > Finished Successfully. Log: {log_file}")

        except subprocess.CalledProcessError:
            print(f"  > FAILED. Check log: {log_file}")
            subprocess.run(["ray", "stop", "--force"])
            continue


if __name__ == "__main__":
    run_sweep()
