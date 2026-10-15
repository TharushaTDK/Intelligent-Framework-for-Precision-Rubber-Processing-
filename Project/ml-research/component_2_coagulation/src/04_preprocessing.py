"""Step 4: Basic preprocessing checks; raw data remains unchanged."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/"data"/"raw"/"latex_coagulation_dataset_final.csv"
OUT=ROOT/"data"/"processed"; REP=ROOT/"outputs"/"reports"
def main():
    if not RAW.exists(): raise FileNotFoundError(RAW)
    OUT.mkdir(parents=True,exist_ok=True); REP.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(RAW)
    df["record_date"]=pd.to_datetime(df["record_date"],errors="coerce")
    df.to_csv(OUT/"04_preprocessed_copy.csv",index=False,date_format="%Y-%m-%d")
    pd.DataFrame([{"rows":len(df),"missing_cells":int(df.isna().sum().sum()),
                   "duplicate_rows":int(df.duplicated().sum()),
                   "note":"Date parsing and copy only; no target values are changed."}]).to_csv(REP/"04_preprocessing_summary.csv",index=False)
    print("Step 4 complete.")
if __name__=="__main__": main()
