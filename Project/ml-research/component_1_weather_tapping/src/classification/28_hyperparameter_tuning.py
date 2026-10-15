from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
subprocess.run([sys.executable,str(ROOT/'src/40_tune_candidate_models.py')],cwd=ROOT,check=True)
