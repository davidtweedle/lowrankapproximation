import os
import subprocess
import itertools

# --- HYPERPARAMETER GRID ---
# Matrix Learning Rates (for Muon and low-rank Body)
# Paper suggests ~0.016. We bracket that.
matrix_lrs = [0.01, 0.015, 0.02]

# Adam Learning Rates (for Embeddings/Heads/etc)
# Paper suggests ~0.003. We bracket that.
adam_lrs = [0.001, 0.003, 0.005]

# --- CONSTANTS ---
LRA_RANK = "32"
LRA_KRYLOV = "0"


def run_sweep():
    # Global Cleanup before starting
    print(">>> Killing old Ray processes...")
    subprocess.run(["ray", "stop", "--force"])

    combinations = list(itertools.product(matrix_lrs, adam_lrs))
    total = len(combinations) * 2

    count = 0
    for i, (mlr, alr) in enumerate(combinations):

        # ==========================================
        # 1. RUN MUON (Linear Schedule)
        # ==========================================
        count += 1
        run_id = f"Muon_M{mlr}_A{alr}_WSD"
        print(f"\n=== [{count}/{total}] STARTING MUON: {run_id} ===")

        env = os.environ.copy()
        env["TUNE_MUON_LR"] = str(mlr)
        env["TUNE_ADAM_LR"] = str(alr)
        env["TUNE_STEPS"] = "4000"
        env["RUN_ID_SUFFIX"] = run_id

        # Force Online Logging
        env["WANDB_MODE"] = "online"
        env["WANDB_ENTITY"] = "david-tweedle-none"
        env["WANDB_PROJECT"] = "lroo"

        # Clean Checkpoints
        subprocess.run(f"rm -rf /workspace/marin_logs/checkpoints/speedrun/muon_{run_id}", shell=True)

        try:
            # Create logs dir
            os.makedirs("sweep_logs", exist_ok=True)

            # Run Muon Script
            with open(f"sweep_logs/{run_id}.log", "w") as f:
                subprocess.run(
                    ["python", "experiments/speedrun/lra_svd_run/run_muon.py"],
                    env=env,
                    check=True,
                    stdout=f,
                    stderr=subprocess.STDOUT
                )
            print("  > Muon Finished Successfully.")

        except subprocess.CalledProcessError:
            print("  > MUON FAILED. Check logs.")
            # Cleanup Ray to prevent OOM for next run
            subprocess.run(["ray", "stop", "--force"])

        # ==========================================
        # 2. RUN LRA (Linear Matrix / Cosine Adam)
        # ==========================================
        count += 1
        run_id = f"LRA_M{mlr}_A{alr}_R{LRA_RANK}_K{LRA_KRYLOV}"
        print(f"\n=== [{count}/{total}] STARTING LRA: {run_id} ===")

        env = os.environ.copy()
        env["TUNE_LRA_LR"] = str(mlr)
        env["TUNE_ADAM_LR"] = str(alr)
        env["TUNE_RANK"] = LRA_RANK
        env["TUNE_KRYLOV"] = LRA_KRYLOV
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
            with open(f"sweep_logs/{run_id}.log", "w") as f:
                subprocess.run(
                    ["python", "experiments/speedrun/lra_svd_run/run.py"],
                    env=env,
                    check=True,
                    stdout=f,
                    stderr=subprocess.STDOUT
                )
            print("  > LRA Finished Successfully.")

        except subprocess.CalledProcessError:
            print("  > LRA FAILED. Check logs.")
            subprocess.run(["ray", "stop", "--force"])


if __name__ == "__main__":
    run_sweep()
