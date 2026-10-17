from common import ensure_dirs, load_raw, clean_basic, SPLITS, REPORTS
from sklearn.model_selection import train_test_split
import pandas as pd
def main():
    ensure_dirs()
    df = clean_basic(load_raw())
    # One deterministic 80/20 holdout per target. Split indices are saved for reproducibility.
    tasks = {
        "drc_prediction": "latex_drc_pct",
        "tapping_advisory": "tapping_suitability_proxy",
        "price_estimation": "price_lkr_per_kg"
    }
    report=[]
    for task, target in tasks.items():
        valid = df[target].notna()
        idx = df.index[valid].to_numpy()
        stratify = df.loc[idx, target] if target == "tapping_suitability_proxy" else None
        train_idx, test_idx = train_test_split(idx, test_size=.2, random_state=340, stratify=stratify)
        pd.DataFrame({"row_index":train_idx}).to_csv(SPLITS/f"{task}_train_indices.csv",index=False)
        pd.DataFrame({"row_index":test_idx}).to_csv(SPLITS/f"{task}_test_indices.csv",index=False)
        report.append(f"{task}: target={target}; usable={len(idx)}; train={len(train_idx)}; test={len(test_idx)}")
    (REPORTS/"04_split_report.txt").write_text("\\n".join(report),encoding="utf-8")
    print("\\n".join(report))
if __name__ == "__main__": main()
