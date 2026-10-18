from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from train_candidate import fit_regression_candidate
if __name__ == "__main__":
    print("STAGE 13: Gradient Boosting regression for DRC and price")
    fit_regression_candidate("gradient_boosting")
