"""Step 5: Exploratory data analysis."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"/"processed"/"latex_coagulation_dataset_final.csv"
REP=ROOT/"outputs"/"reports"; FIG=ROOT/"outputs"/"figures"
COLS=["drc","latex_volume_L","water_amount_L","prepared_1pct_formic_acid_solution_mL","coagulation_time_hours"]
def main():
    if not DATA.exists(): raise FileNotFoundError(DATA)
    REP.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(DATA); cols=[c for c in COLS if c in df]
    df[cols].describe().T.to_csv(REP/"05_eda_statistics.csv")
    df[cols].corr().to_csv(REP/"05_eda_correlations.csv")
    for c in cols:
        plt.figure(); plt.hist(pd.to_numeric(df[c],errors="coerce").dropna(),bins=25)
        plt.title(f"Distribution: {c}"); plt.xlabel(c); plt.ylabel("Frequency"); plt.tight_layout()
        plt.savefig(FIG/f"05_{c}_distribution.png",dpi=140); plt.close()
    print("Step 5 EDA complete.")
if __name__=="__main__": main()
