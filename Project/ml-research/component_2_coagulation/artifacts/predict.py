"""Run both Component 2 models on a CSV of batches.

Usage:
    python predict.py sample_input.csv

The input needs drc and latex_volume_L. Model 1 fills in water and acid,
then model 2 uses those to estimate coagulation time. If the CSV already has
water_amount_L and prepared_1pct_formic_acid_solution_mL, those values are
used for model 2 instead.
"""
from pathlib import Path
import json, sys
import pandas as pd, joblib

HERE=Path(__file__).resolve().parent
water_acid=joblib.load(HERE/"models"/"model_1_water_acid_best.joblib")
coag_time=joblib.load(HERE/"models"/"model_2_coagulation_time_best.joblib")
schema=json.loads((HERE/"input_schema.json").read_text(encoding="utf-8"))

def predict(df):
    out=df.copy()
    wa=water_acid.predict(out[["drc","latex_volume_L"]])
    out["recommended_water_amount_L"]=wa[:,0]
    out["recommended_1pct_formic_acid_mL"]=wa[:,1]
    time_in=pd.DataFrame({
        "drc":out["drc"],"latex_volume_L":out["latex_volume_L"],
        "water_amount_L":out.get("water_amount_L",out["recommended_water_amount_L"]),
        "prepared_1pct_formic_acid_solution_mL":out.get("prepared_1pct_formic_acid_solution_mL",out["recommended_1pct_formic_acid_mL"]),
    })
    out["predicted_coagulation_time_hours"]=coag_time.predict(time_in)
    # flag rows outside the training range, predictions there are less trustworthy
    outside=pd.Series(False,index=out.index)
    for col in ["drc","latex_volume_L"]:
        outside|=(out[col]<schema[col]["min"])|(out[col]>schema[col]["max"])
    out["outside_training_range"]=outside
    return out

if __name__=="__main__":
    path=sys.argv[1] if len(sys.argv)>1 else HERE/"sample_input.csv"
    print(predict(pd.read_csv(path)).to_string(index=False))
