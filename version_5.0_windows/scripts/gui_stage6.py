"""Stage-6 GUI entry: fit current keypoints with the CLI, then open the viewer."""
from pathlib import Path
import subprocess
import sys

import run_roles

SCRIPTS = Path(__file__).resolve().parent


def calculate_and_view(run_dir):
    run_dir = Path(run_dir).resolve()
    roles = run_roles.read(run_dir)
    engine = SCRIPTS / "engine_dapi"
    print("[GUI] Calculating matrix from current keypoints (including manual QC corrections)...", flush=True)
    subprocess.run([
        sys.executable, str(SCRIPTS / "cli_stage5.py"), "--engine", str(engine),
        "--run-dir", str(run_dir), "--pair", roles.pair.name,
        "--apply-manual-centroids",
    ], cwd=engine, check=True)
    # Failure above stops here: never show a stale matrix after a failed fit.
    subprocess.run([sys.executable, str(engine / "6_final_alignment.py"), str(run_dir)],
                   cwd=engine, check=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python gui_stage6.py <RUN_DIR>")
    calculate_and_view(sys.argv[1])
