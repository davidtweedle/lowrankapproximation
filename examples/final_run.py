import os
import subprocess
import time

# --- GLOBAL SETTINGS ---
# 40,000 Steps = ~5.2 Billion Tokens (Approx 1.7x Chinchilla for 150M)
STEPS = 1000
WANDB_PROJECT = "lroo"
ENTITY = "david-tweedle-none"

# --- CONFIGURATIONS ---

config_adam = {
    "name": "AdamW_Baseline",
    "script": "experiments/speedrun/lra_svd_run/run_adam.py",
    "env": {
        "TUNE_ADAM_LR": "0.003",
        "TUNE_ADAM_WD": "0.1",
        "TUNE_STEPS": str(STEPS),
        "RUN_ID_SUFFIX": "test_AdamW_Cosine_40k",
    }
}

config_muon = {
    "name": "Muon_Competitor",
    "script": "experiments/speedrun/lra_svd_run/run_muon.py",
    "env": {
        "TUNE_MUON_LR": "0.015",
        "TUNE_ADAM_LR_RATIO": "0.2",
        "TUNE_KIMI": "False",
        "TUNE_STEPS": str(STEPS),
        "RUN_ID_SUFFIX": "test_Muon_wsd_40k",
    }
}

config_lra = {
    "name": "LRA_Ours",
    "script": "experiments/speedrun/lra_svd_run/run.py",
    "env": {
        "TUNE_LRA_LR": "0.02",
        "TUNE_ADAM_LR_RATIO": "0.2",
        "TUNE_KRYLOV": "0",
        "TUNE_RANK": "32",
        "TUNE_STEPS": str(STEPS),
        "RUN_ID_SUFFIX": "test_lroo_Best_40k",
    }
}

experiments = [config_adam, config_muon, config_lra]


def safe_cleanup():
    """
    Kills Ray and zombie Python processes, but SPARES the current script.
    """
    print(">>> Performing Safe Cleanup...")

    # 1. Stop Ray (Standard cleanup)
    subprocess.run(["ray", "stop", "--force"], stderr=subprocess.DEVNULL)

    # 2. Kill other Python processes (Zombies/Hanging GPU tasks)
    # We grep for python, exclude our own PID, and kill the rest.
    current_pid = os.getpid()

    # Command breakdown:
    # pgrep -f python   : List all python process IDs
    # grep -v {pid}     : Remove our PID from the list
    # xargs -r kill -9  : Kill the remaining IDs (if any exist)
    kill_cmd = f"pgrep -f python | grep -v {current_pid} | xargs -r kill -9"

    try:
        subprocess.run(kill_cmd, shell=True)
    except Exception as e:
        print(f"Warning during process kill: {e}")

    time.sleep(5)  # Give the OS time to reclaim VRAM


def execute_all():
    # Initial cleanup
    safe_cleanup()

    for exp in experiments:
        print("\n" + "="*60)
        print(f"🚀 LAUNCHING: {exp['name']}")
        print(f"   Steps: {STEPS}")
        print(f"   Params: {exp['env']}")
        print("="*60 + "\n")

        # Base Env
        env = os.environ.copy()
        env["WANDB_MODE"] = "online"
        env["WANDB_ENTITY"] = ENTITY
        env["WANDB_PROJECT"] = WANDB_PROJECT
        env["WANDB_TAGS"] = f"final,40k,{exp['name']}"

        # Inject Specific Params
        env.update(exp['env'])

        # Cleanup previous checkpoint
        subprocess.run(f"rm -rf /workspace/marin_logs/checkpoints/speedrun/{env['RUN_ID_SUFFIX']}", shell=True)

        try:
            # Run!
            subprocess.run(
                ["python", exp["script"]],
                env=env,
                check=True
            )
            print(f"✅ {exp['name']} Completed Successfully.")

        except subprocess.CalledProcessError:
            print(f"❌ {exp['name']} FAILED/CRASHED.")

        # Cleanup between runs
        safe_cleanup()


if __name__ == "__main__":
    execute_all()
