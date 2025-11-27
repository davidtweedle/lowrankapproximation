import os
import subprocess
import itertools

# --- SHARED GRID (Paper Aligned) ---
matrix_lrs = [0.01, 0.015, 0.02]      # Muon & LRA Main LR
adam_lrs = [0.001, 0.003, 0.005]      # Embed/Head LR

def run_sweep():
    # Initial Cleanup
    subprocess.run(["ray", "stop", "--force"])

    combinations = list(itertools.product(matrix_lrs, adam_lrs))
    print(f"--- Launching Combined Sweep ({len(combinations)} Configs x 2 Optimizers) ---")

    for i, (mlr, alr) in enumerate(combinations):
        print(f"\n=== [{i+1}/{len(combinations)}] Config: Matrix={mlr}, Adam={alr} ===")

        # -----------------------
        # 1. Run MUON
        # -----------------------
        muon_id = f"Muon_M{mlr}_A{alr}"
        print(f"  > Launching: {muon_id}")

        env_muon = os.environ.copy()
        env_muon["TUNE_MUON_LR"] = str(mlr)
        env_muon["TUNE_ADAM_LR"] = str(alr)
        env_muon["RUN_ID_SUFFIX"] = muon_id
        env_muon["WANDB_MODE"] = "online"
        env_muon["WANDB_ENTITY"] = "david-tweedle-none"

        # Cleanup & Run Muon
        subprocess.run(f"rm -rf /workspace/marin_logs/checkpoints/speedrun/muon_lin_{muon_id}", shell=True)
        try:
            with open(f"sweep_logs/{muon_id}.log", "w") as f:
                subprocess.run(
                    ["python", "experiments/speedrun/lra_svd_run/run_muon.py"],
                    env=env_muon, check=True, stdout=f, stderr=subprocess.STDOUT
                )
        except subprocess.CalledProcessError:
            print(f"    X Muon Failed.")
            subprocess.run(["ray", "stop", "--force"])

        # -----------------------
        # 2. Run LRA (Rank 32, K=0)
        # -----------------------
        lra_id = f"LRA_M{mlr}_A{alr}"
        print(f"  > Launching: {lra_id}")

        env_lra = os.environ.copy()
        env_lra["TUNE_LRA_LR"] = str(mlr)
        env_lra["TUNE_ADAM_LR"] = str(alr)
        env_lra["RUN_ID_SUFFIX"] = lra_id
        # Fixed params
        env_lra["TUNE_KRYLOV"] = "0"
        env_lra["TUNE_RANK"] = "32"
        env_lra["WANDB_MODE"] = "online"
        env_lra["WANDB_ENTITY"] = "david-tweedle-none"

        # Cleanup & Run LRA
        subprocess.run(f"rm -rf /workspace/marin_logs/checkpoints/speedrun/lra_{lra_id}", shell=True)
        try:
            with open(f"sweep_logs/{lra_id}.log", "w") as f:
                subprocess.run(
                    ["python", "experiments/speedrun/lra_svd_run/run.py"],
                    env=env_lra, check=True, stdout=f, stderr=subprocess.STDOUT
                )
        except subprocess.CalledProcessError:
            print("    X LRA Failed.")
            subprocess.run(["ray", "stop", "--force"])


if __name__ == "__main__":
    os.makedirs("sweep_logs", exist_ok=True)
    run_sweep()
