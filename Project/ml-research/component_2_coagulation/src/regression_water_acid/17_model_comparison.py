"""Select best regression algorithm for water + prepared 1% formic acid solution."""
from pathlib import Path
import json, sys
import pandas as pd, joblib
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _candidate_models import candidates
ROOT=Path(__file__).resolve().parents[2]
TRAIN=ROOT/"data"/"splits"/"Component2_Train.csv"
MODELS=ROOT/"outputs"/"models"; REPORTS=ROOT/"outputs"/"reports"
FEATURES=["drc","latex_volume_L"]
TARGETS=["water_amount_L","prepared_1pct_formic_acid_solution_mL"]
def main():
    if not TRAIN.exists(): raise FileNotFoundError("Run src/09_batch_aware_split.py first.")
    MODELS.mkdir(parents=True,exist_ok=True); REPORTS.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(TRAIN); X=df[FEATURES]; y=df[TARGETS]
    cv=KFold(n_splits=5,shuffle=True,random_state=42); rows=[]; ranked=[]
    for name,est in candidates().items():
        pipe=Pipeline([("imputer",SimpleImputer(strategy="median")),("model",MultiOutputRegressor(est))])
        scores=cross_val_score(pipe,X,y,cv=cv,scoring="neg_mean_absolute_error",n_jobs=1)
        rows.append({"algorithm":name,"mean_CV_MAE":-scores.mean(),"std_CV_MAE":scores.std()})
        ranked.append((-scores.mean(),name,pipe))
    ranked.sort(key=lambda x:x[0]); negscore,name,model=ranked[0]; model.fit(X,y)
    joblib.dump(model,MODELS/"model_1_water_acid_best.joblib")
    pd.DataFrame(rows).sort_values("mean_CV_MAE").to_csv(REPORTS/"water_acid_algorithm_comparison.csv",index=False)
    (REPORTS/"water_acid_selected_model.json").write_text(json.dumps({"selected_algorithm":name,"targets":TARGETS,"cv_mae":-negscore},indent=2),encoding="utf-8")
    print("Selected water/acid model:",name)
if __name__=="__main__": main()
