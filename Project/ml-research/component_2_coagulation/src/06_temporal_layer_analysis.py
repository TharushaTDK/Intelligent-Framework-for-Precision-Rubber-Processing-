"""Step 6: Descriptive date/time layer analysis (not a time-series forecast)."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"/"processed"/"latex_coagulation_dataset_final.csv"
REP=ROOT/"outputs"/"reports"
def main():
    if not DATA.exists(): raise FileNotFoundError(DATA)
    REP.mkdir(parents=True,exist_ok=True); df=pd.read_csv(DATA)
    df["record_date"]=pd.to_datetime(df["record_date"],errors="coerce")
    df["record_month"]=df["record_date"].dt.to_period("M").astype(str)
    target=[c for c in ["water_amount_L","prepared_1pct_formic_acid_solution_mL","coagulation_time_minutes"] if c in df]
    df.groupby("record_month")[target].agg(["count","mean","min","max"]).to_csv(REP/"06_temporal_summary.csv")
    print("Step 6 temporal summary complete.")
if __name__=="__main__": main()
