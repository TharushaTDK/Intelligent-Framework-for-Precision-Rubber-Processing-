"""Step 30: Cross-validation algorithm comparison for two model tasks."""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import make_scorer, mean_absolute_error
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"/"processed"/"07_feature_engineered.csv"; REP=ROOT/"outputs"/"reports"
FEATURES=["drc","latex_volume_L"]
FEATURES_TIME=FEATURES+["water_amount_L","prepared_1pct_formic_acid_solution_mL"]
WATER_ACID=["water_amount_L","prepared_1pct_formic_acid_solution_mL"]
TIME=["coagulation_time_minutes"]
def algos():
    return {
      "LinearRegression":LinearRegression(),
      "Ridge":Ridge(alpha=1.0),
      "RandomForest":RandomForestRegressor(n_estimators=200,min_samples_leaf=2,random_state=42,n_jobs=-1),
      "ExtraTrees":ExtraTreesRegressor(n_estimators=200,min_samples_leaf=2,random_state=42,n_jobs=-1),
      "HistGradientBoosting":HistGradientBoostingRegressor(random_state=42),
    }
def main():
    if not DATA.exists(): raise FileNotFoundError("Run step 7 first.")
    REP.mkdir(parents=True,exist_ok=True); df=pd.read_csv(DATA)
    cv=KFold(n_splits=5,shuffle=True,random_state=42); rows=[]
    for task,targets,features in [("model_1_water_acid",WATER_ACID,FEATURES),("model_2_coagulation_time",TIME,FEATURES_TIME)]:
      X=df[features]; y=df[targets] if len(targets)>1 else df[targets[0]]
      for name,est in algos().items():
        model=MultiOutputRegressor(est) if len(targets)>1 else est
        pipe=Pipeline([("imputer",SimpleImputer(strategy="median")),("model",model)])
        scores=cross_val_score(pipe,X,y,cv=cv,scoring="neg_mean_absolute_error",n_jobs=1)
        rows.append({"task":task,"algorithm":name,"mean_CV_MAE":-scores.mean(),"std_CV_MAE":scores.std(),
                     "note":"Synthetic data: CV is pipeline comparison only, not real-world validation."})
    pd.DataFrame(rows).to_csv(REP/"30_cross_validation_comparison.csv",index=False)
    print("Step 30 CV comparison complete.")
if __name__=="__main__": main()
