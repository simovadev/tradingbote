"""Download v19_best.pt + recap depuis Vast vers PC local."""
import os
import subprocess
import sys
from pathlib import Path

VAST_HOST = "99.27.206.243"
VAST_PORT = 61443
KEY_PATH = os.path.expanduser("~/.ssh/vast_v8")
ROOT = "C:/Users/Shadow/TradingBot" if sys.platform == "win32" else os.path.expanduser("~/TradingBot")


def scp(remote_path: str, local_path: str):
    cmd = [
        "scp", "-i", KEY_PATH, "-P", str(VAST_PORT),
        "-o", "StrictHostKeyChecking=no",
        f"root@{VAST_HOST}:{remote_path}",
        local_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0:
        size = Path(local_path).stat().st_size / 1024
        print(f"  OK {local_path} ({size:.1f} KB)")
        return True
    print(f"  FAIL {remote_path} : {r.stderr.strip()}")
    return False


def main():
    print("=== Download V19 model + scripts depuis Vast ===\n")

    Path(f"{ROOT}/bot_v2").mkdir(parents=True, exist_ok=True)

    targets = [
        ("/workspace/TradingBot/v19_best.pt", f"{ROOT}/bot_v2/v19_best.pt"),
        ("/workspace/TradingBot/v19_train_recap.json", f"{ROOT}/v19_train_recap.json"),
        ("/workspace/TradingBot/v19_adversarial_recap.json", f"{ROOT}/v19_adversarial_recap.json"),
    ]

    for remote, local in targets:
        scp(remote, local)

    # Verify model loadable
    print("\n=== Test load V19 ===")
    try:
        sys.path.insert(0, ROOT)
        from bot_v2.v19_inference import V19Predictor
        p = V19Predictor.get_instance(Path(f"{ROOT}/bot_v2/v19_best.pt"))
        if p is None:
            print("FAIL : V19Predictor.get_instance returned None")
        else:
            print(f"OK : model loaded sur {p.device}")
            print(f"  n_assets={p.n_assets} n_ict={p.n_ict}")
    except Exception as e:
        print(f"FAIL : {e}")


if __name__ == "__main__":
    main()
