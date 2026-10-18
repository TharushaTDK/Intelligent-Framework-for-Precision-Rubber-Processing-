from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
subprocess.run([sys.executable,str(ROOT/'src/classification/21_compare_classification_models.py')],cwd=ROOT,check=True)
