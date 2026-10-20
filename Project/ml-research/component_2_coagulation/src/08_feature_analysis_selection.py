"""Step 8: Feature correlation and selection review; does not automatically remove features."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"/"processed"/"07_feature_engineered.csv"; REP=ROOT/"outputs"/"reports"
FEATURES=["drc","latex_volume_L"]
TARGETS=["water_amount_L","prepared_1pct_formic_acid_solution_mL","coagulation_time_minutes"]
def main():
    if not DATA.exists(): raise FileNotFoundError("Run step 7 first.")
    REP.mkdir(parents=True,exist_ok=True); df=pd.read_csv(DATA)
    df[FEATURES+TARGETS].corr().loc[FEATURES,TARGETS].to_csv(REP/"08_feature_target_correlations.csv")
    pd.DataFrame({"feature":FEATURES,"decision":["retain for baseline comparison"]*len(FEATURES)}).to_csv(REP/"08_feature_selection_review.csv",index=False)
    print("Step 8 complete; review correlations before changing features.")
if __name__=="__main__": main()
