"""Evaluate the two separately selected models on the held-out test set."""
from pathlib import Path
import pandas as pd, joblib
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
ROOT=Path(__file__).resolve().parents[1]; SPLITS=ROOT/"data"/"splits"; MODELS=ROOT/"outputs"/"models"; REP=ROOT/"outputs"/"reports"
FEATURES=["drc","latex_volume_L"]
FEATURES_TIME=FEATURES+["water_amount_L","prepared_1pct_formic_acid_solution_mL"]
TASKS=[("model_1_water_acid_best.joblib",["water_amount_L","prepared_1pct_formic_acid_solution_mL"],FEATURES),
       ("model_2_coagulation_time_best.joblib",["coagulation_time_minutes"],FEATURES_TIME)]
def main():
    path=SPLITS/"Component2_Test.csv"
    if not path.exists(): raise FileNotFoundError("Run 09_batch_aware_split.py first.")
    REP.mkdir(parents=True,exist_ok=True); test=pd.read_csv(path); rows=[]; out=test[["batch_id","record_date"]+FEATURES].copy()
    for filename,targets,features in TASKS:
        model_path=MODELS/filename
        if not model_path.exists(): raise FileNotFoundError(f"Train both models first; missing {model_path}")
        model=joblib.load(model_path); pred=model.predict(test[features])
        if len(targets)==1: pred=pred.reshape(-1,1)
        for i,target in enumerate(targets):
            y=test[target].to_numpy(); yp=pred[:,i]
            rows.append({"target":target,"MAE":mean_absolute_error(y,yp),"RMSE":mean_squared_error(y,yp)**.5,"R2":r2_score(y,yp)})
            out[f"actual_{target}"]=y; out[f"predicted_{target}"]=yp
    pd.DataFrame(rows).to_csv(REP/"30_final_test_metrics.csv",index=False); out.to_csv(REP/"30_test_predictions_vs_actual.csv",index=False)
    print("Saved final test evaluation.")
if __name__=="__main__": main()
