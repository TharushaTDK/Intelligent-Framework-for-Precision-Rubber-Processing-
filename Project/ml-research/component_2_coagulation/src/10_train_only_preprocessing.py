"""Step 10: Fit imputation/scaling only on training data and save the transformers."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"/"splits"; MODELS=ROOT/"outputs"/"models"; REP=ROOT/"outputs"/"reports"
FEATURES=["drc","latex_volume_L"]
def main():
    train_path=DATA/"Component2_Train.csv"
    if not train_path.exists(): raise FileNotFoundError("Run step 9 first.")
    MODELS.mkdir(parents=True,exist_ok=True); REP.mkdir(parents=True,exist_ok=True)
    train=pd.read_csv(train_path); X=train[FEATURES]
    imputer=SimpleImputer(strategy="median"); scaler=StandardScaler()
    X_imp=imputer.fit_transform(X); scaler.fit(X_imp)
    joblib.dump(imputer,MODELS/"10_feature_imputer.joblib"); joblib.dump(scaler,MODELS/"10_feature_scaler.joblib")
    pd.DataFrame({"feature":FEATURES,"imputation_median":imputer.statistics_}).to_csv(REP/"10_train_only_preprocessing.csv",index=False)
    print("Step 10 complete.")
if __name__=="__main__": main()
