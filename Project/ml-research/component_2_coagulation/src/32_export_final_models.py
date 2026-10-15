"""Export exactly two selected model artifacts."""
from pathlib import Path
import json,shutil
ROOT=Path(__file__).resolve().parents[1]; MODELS=ROOT/"outputs"/"models"; OUT=ROOT/"outputs"/"final_models"
TASKS={"model_1_water_acid_best.joblib":"Water and prepared 1% formic acid solution recommendation","model_2_coagulation_time_best.joblib":"Coagulation time prediction"}
def main():
    OUT.mkdir(parents=True,exist_ok=True); manifest={}
    for filename,description in TASKS.items():
        p=MODELS/filename
        if not p.exists(): raise FileNotFoundError(f"Missing {p}; train both models first.")
        shutil.copy2(p,OUT/filename); manifest[filename]={"description":description}
    (OUT/"model_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print("Exported two selected models.")
if __name__=="__main__": main()
