"""Create train, validation, and test datasets in data/splits/."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"/"processed"/"07_feature_engineered.csv"
SPLITS=ROOT/"data"/"splits"; READY=ROOT/"data"/"model_ready"; REPORTS=ROOT/"outputs"/"reports"
FEATURES=["drc","latex_volume_L"]
def main():
    if not DATA.exists(): raise FileNotFoundError("Run src/07_feature_engineering.py first.")
    SPLITS.mkdir(parents=True,exist_ok=True); READY.mkdir(parents=True,exist_ok=True); REPORTS.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(DATA)
    if df["batch_id"].duplicated().any(): raise ValueError("Duplicate batch IDs; resolve before splitting.")
    train_ids,remain=train_test_split(df["batch_id"],test_size=.30,random_state=42)
    val_ids,test_ids=train_test_split(remain,test_size=.50,random_state=42)
    sets={"Train":df[df.batch_id.isin(set(train_ids))].copy(),"Validation":df[df.batch_id.isin(set(val_ids))].copy(),"Test":df[df.batch_id.isin(set(test_ids))].copy()}
    for name,part in sets.items():
        part.to_csv(SPLITS/f"Component2_{name}.csv",index=False)
        part[FEATURES].to_csv(READY/f"Component2_X_{name}.csv",index=False)
        part[["water_amount_L","prepared_1pct_formic_acid_solution_mL"]].to_csv(READY/f"Component2_y_water_acid_{name}.csv",index=False)
        part[["coagulation_time_hours"]].to_csv(READY/f"Component2_y_time_{name}.csv",index=False)
    pd.DataFrame([{"train_rows":len(sets["Train"]),"validation_rows":len(sets["Validation"]),"test_rows":len(sets["Test"])}]).to_csv(REPORTS/"09_split_summary.csv",index=False)
    print("Created train, validation, test and model_ready files.")
if __name__=="__main__": main()
