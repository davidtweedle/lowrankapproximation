import os
import subprocess
import itertools
import time

# --- GRID ---
muon_lrs = [0.01, 0.02, 0.05]
adam_lrs = [3e-4, 6e-4, 1e-3]

def run_sweep():
    # Ensure clean start
    subprocess.run(["ray", "stop", "--force"])

    combinations = list(itertools.product(muon_lrs, adam_lrs))
    print(f"Total Muon Runs: {len(combinations)}")

    for i, (mlr, alr) in enumerate(combinations):
        run_id = f"M{mlr}_A{alr}"
        print(f"\n[{i+1}/{len(combinations)}] Starting: {run_id}")

        # Pass params
        env = os.environ.copy()
        env["TUNE_MUON_LR"] = str(mlr)
        env["TUNE_ADAM_LR"] = str(alr)
        env["RUN_ID_SUFFIX"] = run_id

        env["WANDB_MODE"] = "online"
        env["WANDB_ENTITY"] = "david-tweedle-none"
        env["WANDB_PROJECT"] = "lra-optimizer"

        # Clean up previous run with same ID
        subprocess.run(f"rm -rf /workspace/marin_logs/checkpoints/speedrun/muon_{run_id}", shell=True)

        try:
            log_file = f"sweep_logs/muon_{run_id}.log"
            os.makedirs("sweep_logs", exist_ok=True)

            with open(log_file, "w") as f:
                subprocess.run(
                    ["python", "experiments/speedrun/lra_svd_run/run_muon.py"],
                    env=env,
                    check=True,
                    stdout=f,
                    stderr=subprocess.STDOUT
                )
            print(f"  > Finished Successfully. Log: {log_file}")

        except subprocess.CalledProcessError:
            print(f"  > FAILED. Check log: {log_file}")
            # Restart Ray on failure
            subprocess.run(["ray", "stop", "--force"])
            continue

if __name__ == "__main__":
    run_sweep()
