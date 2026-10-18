from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from train_candidate import fit_classification_candidate
if __name__ == "__main__":
    print("STAGE 24: Extra Trees tapping advisory classifier")
    fit_classification_candidate("extra_trees")
