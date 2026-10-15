"""Optional SHAP explanations for both selected regression models."""
from pathlib import Path
import pandas as pd, joblib
ROOT=Path(__file__).resolve().parents[1]; TRAIN=ROOT/"data"/"splits"/"Component2_Train.csv"; MODELS=ROOT/"outputs"/"models"; REP=ROOT/"outputs"/"reports"
FEATURES=["drc","latex_volume_L"]
FEATURES_TIME=FEATURES+["water_amount_L","prepared_1pct_formic_acid_solution_mL"]
TASKS=[("model_1_water_acid_best.joblib",FEATURES),("model_2_coagulation_time_best.joblib",FEATURES_TIME)]
def main():
    try: import shap
    except ImportError: print("SHAP not installed; skipping optional explainability."); return
    if not TRAIN.exists(): raise FileNotFoundError("Run 09_batch_aware_split.py first.")
    REP.mkdir(parents=True,exist_ok=True); train=pd.read_csv(TRAIN); rows=[]
    for fn,features in TASKS:
        path=MODELS/fn
        if not path.exists(): print(f"Skipping missing model {fn}"); continue
        model=joblib.load(path); X=train[features].sample(min(100,len(train)),random_state=42)
        try:
            import numpy as np
            exp=shap.Explainer(model.predict,X)(X).values
            arr=np.asarray(exp)
            if arr.ndim==3: vals=np.mean(np.abs(arr),axis=(0,2))
            else: vals=np.mean(np.abs(arr),axis=0)
            for feature,value in zip(features,vals): rows.append({"model":fn,"feature":feature,"mean_absolute_shap":float(value)})
        except Exception as e: rows.append({"model":fn,"feature":"ERROR","mean_absolute_shap":None,"details":str(e)})
    pd.DataFrame(rows).to_csv(REP/"31_shap_feature_importance.csv",index=False)
    print("SHAP report saved.")
if __name__=="__main__": main()
