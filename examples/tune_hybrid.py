import os
import subprocess
import itertools

# --- GRID ---
# 1. Matrix LR: Trying around 0.02 (current baseline)
lra_lrs = [0.01, 0.02, 0.04]

# 2. Adam LR: Standard range for 150M model
adam_lrs = [3e-4, 6e-4, 1e-3]

# 3. Krylov: 0 = Random Projection, 1 = 1 Power Iteration
krylov_iters = [0, 1]

def run_sweep():
    # 1. Clear any old Ray processes first
    subprocess.run(["ray", "stop", "--force"])

    combinations = list(itertools.product(lra_lrs, adam_lrs, krylov_iters))
    print(f"Total Runs: {len(combinations)}")

    for i, (lra, adam, k) in enumerate(combinations):
        run_id = f"L{lra}_A{adam}_K{k}"
        print(f"\n[{i+1}/{len(combinations)}] Starting: {run_id}")

        # Pass params
        env = os.environ.copy()
        env["TUNE_LRA_LR"] = str(lra)
        env["TUNE_ADAM_LR"] = str(adam)
        env["TUNE_KRYLOV"] = str(k)
        env["RUN_ID_SUFFIX"] = run_id
        # Ensure WandB is offline
        env["WANDB_MODE"] = "offline"
        try:
            # Run (check=True ensures we catch failures)
            # We redirect stderr/stdout to a log file so your terminal isn't flooded
            log_file = f"sweep_logs/{run_id}.log"
            os.makedirs("sweep_logs", exist_ok=True)
            with open(log_file, "w") as f:
                subprocess.run(
                    ["python", "experiments/speedrun/lra_svd_run/run.py"],
                    env=env,
                    check=True,
                    stdout=f,
                    stderr=subprocess.STDOUT
                )
            print(f"  > Finished Successfully. Log: {log_file}")

        except subprocess.CalledProcessError:
            print(f"  > FAILED. Check log: {log_file}")
            # Restart Ray just in case the failure corrupted memory
            subprocess.run(["ray", "stop", "--force"])
            continue


if __name__ == "__main__":
    run_sweep()
