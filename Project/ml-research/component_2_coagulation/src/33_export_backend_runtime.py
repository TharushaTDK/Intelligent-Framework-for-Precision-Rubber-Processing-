"""Re-export the two selected models for the shared backend's scikit-learn version.

The research artifacts were saved with scikit-learn 1.7.2 and the shared backend runs 1.9.1,
which can't run the old pickles. This clones the exact selected pipelines (same steps,
hyperparameters and random_state), refits them on the same training split and checks the
test predictions against artifacts/metrics/test_predictions_vs_actual.csv before saving.
"""
from pathlib import Path
import json, hashlib, warnings
import numpy as np, pandas as pd, joblib, sklearn
from sklearn.base import clone

ROOT=Path(__file__).resolve().parents[1]
SPLITS=ROOT/"data"/"splits"; ART=ROOT/"artifacts"
RUNTIME=ROOT.parents[1]/"backend"/"app"/"ml"/"component_2_coagulation"
FEATURES=["drc","latex_volume_L"]
FEATURES_TIME=FEATURES+["water_amount_L","prepared_1pct_formic_acid_solution_mL"]
TASKS=[
  ("model_1_water_acid_best.joblib","water_acid_pipeline.joblib",FEATURES,["water_amount_L","prepared_1pct_formic_acid_solution_mL"]),
  ("model_2_coagulation_time_best.joblib","coagulation_time_pipeline.joblib",FEATURES_TIME,["coagulation_time_minutes"]),
]
TOLERANCE=1e-6

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""): h.update(chunk)
    return h.hexdigest()

def main():
    train=pd.read_csv(SPLITS/"Component2_Train.csv"); test=pd.read_csv(SPLITS/"Component2_Test.csv")
    ref=pd.read_csv(ART/"metrics"/"test_predictions_vs_actual.csv")
    if not (ref["batch_id"].values==test["batch_id"].values).all(): raise ValueError("Test split and saved predictions are not aligned.")
    report={"scikit-learn":sklearn.__version__,"tolerance":TOLERANCE,"models":{}}
    for research_file,runtime_file,features,targets in TASKS:
        # loading the old pickle is only used to copy its unfitted configuration
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            old=joblib.load(ART/"models"/research_file)
        model=clone(old)
        y=train[targets] if len(targets)>1 else train[targets[0]]
        model.fit(train[features],y)
        pred=np.asarray(model.predict(test[features])).reshape(len(test),-1)
        diffs={t:float(np.abs(pred[:,i]-ref[f"predicted_{t}"].to_numpy()).max()) for i,t in enumerate(targets)}
        worst=max(diffs.values())
        if worst>TOLERANCE: raise ValueError(f"{research_file}: refit predictions differ from research by {worst}, not saving.")
        out=RUNTIME/runtime_file
        joblib.dump(model,out)
        report["models"][runtime_file]={"from":research_file,"max_abs_test_diff":diffs,"sha256":sha256(out)}
        print(f"{runtime_file}: max diff {worst:.2e}, saved")
    print(json.dumps(report,indent=2))

if __name__=="__main__": main()
