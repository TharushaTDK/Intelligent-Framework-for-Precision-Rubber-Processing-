"""Step 7: Create a modeling table with simple, transparent features."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"/"processed"/"latex_coagulation_dataset_final.csv"
OUT=ROOT/"data"/"processed"; REP=ROOT/"outputs"/"reports"
FEATURES=["drc","latex_volume_L"]
TARGETS=["water_amount_L","prepared_1pct_formic_acid_solution_mL","coagulation_time_minutes"]
def main():
    if not DATA.exists(): raise FileNotFoundError(DATA)
    OUT.mkdir(parents=True,exist_ok=True); REP.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(DATA)
    needed=["batch_id","record_date"]+FEATURES+TARGETS
    missing=[c for c in needed if c not in df]
    if missing: raise ValueError(f"Missing columns: {missing}")
    df=df[needed].copy()
    for c in FEATURES+TARGETS: df[c]=pd.to_numeric(df[c],errors="coerce")
    df["record_date"]=pd.to_datetime(df["record_date"],errors="coerce")
    # Date-derived feature is deliberately omitted from model inputs to avoid unnecessary temporal leakage.
    clean=df.dropna(subset=FEATURES+TARGETS).copy()
    clean["record_date"]=clean["record_date"].dt.strftime("%Y-%m-%d")
    clean.to_csv(OUT/"07_feature_engineered.csv",index=False)
    pd.DataFrame([{"input_rows":len(df),"output_rows":len(clean),"excluded_rows":len(df)-len(clean),
                   "features":", ".join(FEATURES),"note":"No synthetic target values are altered."}]).to_csv(REP/"07_feature_engineering_summary.csv",index=False)
    print("Step 7 feature table saved.")
if __name__=="__main__": main()
