"""Package the two selected models and everything needed to use them into artifacts/."""
from pathlib import Path
from datetime import date
import json, shutil, hashlib, platform
import numpy as np, pandas as pd, joblib, sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT=Path(__file__).resolve().parents[1]
MODELS=ROOT/"outputs"/"models"; REP=ROOT/"outputs"/"reports"; SPLITS=ROOT/"data"/"splits"
ART=ROOT/"artifacts"
FEATURES=["drc","latex_volume_L"]
FEATURES_TIME=FEATURES+["water_amount_L","prepared_1pct_formic_acid_solution_mL"]
TASKS={
  "model_1_water_acid_best.joblib":{"description":"Water and prepared 1% formic acid solution recommendation",
      "features":FEATURES,"targets":["water_amount_L","prepared_1pct_formic_acid_solution_mL"],"selection":"water_acid_selected_model.json"},
  "model_2_coagulation_time_best.joblib":{"description":"Coagulation time prediction",
      "features":FEATURES_TIME,"targets":["coagulation_time_hours"],"selection":"coagulation_time_selected_model.json"},
}

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""): h.update(chunk)
    return h.hexdigest()

def score(y,yp):
    # MAPE is left out because some targets can be close to zero
    return {"MAE":float(mean_absolute_error(y,yp)),"RMSE":float(mean_squared_error(y,yp)**.5),"R2":float(r2_score(y,yp))}

def main():
    for sub in ["models","metrics","plots"]: (ART/sub).mkdir(parents=True,exist_ok=True)
    train=pd.read_csv(SPLITS/"Component2_Train.csv"); val=pd.read_csv(SPLITS/"Component2_Validation.csv"); test=pd.read_csv(SPLITS/"Component2_Test.csv")
    card={"component":"Component 2 - latex coagulation","exported_on":str(date.today()),
          "environment":{"python":platform.python_version(),"scikit-learn":sklearn.__version__,"pandas":pd.__version__,"numpy":np.__version__,"joblib":joblib.__version__},
          "data":{"train_rows":len(train),"validation_rows":len(val),"test_rows":len(test),
                  "train_file_sha256":sha256(SPLITS/"Component2_Train.csv")},
          "models":{}}
    metric_rows=[]; pred_out=test[["batch_id"]].copy()
    for filename,task in TASKS.items():
        src=MODELS/filename
        if not src.exists(): raise FileNotFoundError(f"Missing {src}; train both models first.")
        shutil.copy2(src,ART/"models"/filename)
        model=joblib.load(src); sel=json.loads((REP/task["selection"]).read_text(encoding="utf-8"))
        entry={"description":task["description"],"algorithm":sel["selected_algorithm"],
               "input_features":task["features"],"targets":task["targets"],
               "cv_mae_train_5fold":abs(sel["cv_mae"]),"file_sha256":sha256(src),"metrics":{}}
        for split_name,df in [("validation",val),("test",test)]:
            pred=np.asarray(model.predict(df[task["features"]])).reshape(len(df),-1)
            for i,t in enumerate(task["targets"]):
                m=score(df[t],pred[:,i]); entry["metrics"].setdefault(split_name,{})[t]=m
                metric_rows.append({"model":filename,"split":split_name,"target":t,**m})
                if split_name=="test": pred_out[f"actual_{t}"]=df[t].values; pred_out[f"predicted_{t}"]=pred[:,i]
        card["models"][filename]=entry

    # input ranges seen in training, so callers can tell when they are outside what the model knows
    schema={}
    for col in FEATURES_TIME:
        s=train[col]; schema[col]={"type":"float","min":float(s.min()),"max":float(s.max()),"mean":float(s.mean()),"median":float(s.median())}
    (ART/"input_schema.json").write_text(json.dumps(schema,indent=2),encoding="utf-8")
    (ART/"model_card.json").write_text(json.dumps(card,indent=2),encoding="utf-8")
    pd.DataFrame(metric_rows).to_csv(ART/"metrics"/"metrics_validation_and_test.csv",index=False)
    pred_out.to_csv(ART/"metrics"/"test_predictions_vs_actual.csv",index=False)
    shutil.copy2(REP/"water_acid_algorithm_comparison.csv",ART/"metrics"/"water_acid_algorithm_comparison.csv")
    shutil.copy2(REP/"coagulation_time_algorithm_comparison.csv",ART/"metrics"/"coagulation_time_algorithm_comparison.csv")
    if (REP/"31_shap_feature_importance.csv").exists(): shutil.copy2(REP/"31_shap_feature_importance.csv",ART/"metrics"/"shap_feature_importance.csv")

    targets=TASKS["model_1_water_acid_best.joblib"]["targets"]+TASKS["model_2_coagulation_time_best.joblib"]["targets"]
    for t in targets:
        y=pred_out[f"actual_{t}"]; yp=pred_out[f"predicted_{t}"]; r2=r2_score(y,yp)
        fig,ax=plt.subplots(1,2,figsize=(10,4))
        ax[0].scatter(y,yp,s=10,alpha=.6); lo,hi=min(y.min(),yp.min()),max(y.max(),yp.max())
        ax[0].plot([lo,hi],[lo,hi],"k--",lw=1); ax[0].set_xlabel("actual"); ax[0].set_ylabel("predicted"); ax[0].set_title(f"{t} (test R2 = {r2:.3f})")
        ax[1].hist(y-yp,bins=30); ax[1].axvline(0,color="k",lw=1); ax[1].set_xlabel("actual - predicted"); ax[1].set_title("residuals")
        fig.tight_layout(); fig.savefig(ART/"plots"/f"test_{t}.png",dpi=120); plt.close(fig)
    if (REP/"31_shap_feature_importance.csv").exists():
        shap_df=pd.read_csv(REP/"31_shap_feature_importance.csv").dropna(subset=["mean_absolute_shap"])
        for fn,grp in shap_df.groupby("model"):
            grp=grp.sort_values("mean_absolute_shap"); fig,ax=plt.subplots(figsize=(6,3))
            ax.barh(grp["feature"],grp["mean_absolute_shap"]); ax.set_xlabel("mean |SHAP value|"); ax.set_title(fn.replace(".joblib",""))
            fig.tight_layout(); fig.savefig(ART/"plots"/f"shap_{fn.replace('_best.joblib','')}.png",dpi=120); plt.close(fig)

    # small sample so anyone can try the models straight away
    test[FEATURES_TIME].head(5).to_csv(ART/"sample_input.csv",index=False)
    print("Artifacts written to",ART)

if __name__=="__main__": main()
