"""Stage 30 (data/split integrity + leakage audit).

Verifies the stored splits, index disjointness, and the estate-overlap
characteristic of the row-wise split. Writes a machine-readable CSV and a
human-readable text report. No model is fitted here.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    SPLITS, REPORTS, REG_TASKS, CLS_TASK, ensure_dirs, load_raw, clean_basic, sha256_file,
)

TASKS = list(REG_TASKS) + [CLS_TASK]


def main():
    ensure_dirs()
    df = clean_basic(load_raw())

    file_rows = []
    for p in sorted(SPLITS.glob("*_indices.csv")):
        d = pd.read_csv(p)
        file_rows.append({"file": p.name, "rows": len(d), "columns": len(d.columns),
                          "unique_row_index": int(d["row_index"].nunique()),
                          "duplicate_row_index": int(d["row_index"].duplicated().sum()),
                          "sha256": sha256_file(p)})
    pd.DataFrame(file_rows).to_csv(REPORTS / "30_split_file_integrity.csv", index=False)

    overlap_rows = []
    lines = ["SPLIT INTEGRITY & LEAKAGE AUDIT", "=" * 32, ""]
    for task in TASKS:
        tr = set(pd.read_csv(SPLITS / f"{task}_train_indices.csv")["row_index"])
        te = set(pd.read_csv(SPLITS / f"{task}_test_indices.csv")["row_index"])
        inter = tr & te
        tr_est = set(df.loc[df.index.isin(tr), "estate_id"].astype(str))
        te_est = set(df.loc[df.index.isin(te), "estate_id"].astype(str))
        shared = te_est & tr_est
        row = {
            "task": task, "train_rows": len(tr), "test_rows": len(te),
            "index_overlap": len(inter),
            "train_estates": len(tr_est), "test_estates": len(te_est),
            "test_estates_also_in_train": len(shared),
            "test_estates_unseen": len(te_est - tr_est),
            "row_wise_split": True,
        }
        overlap_rows.append(row)
        lines += [
            f"[{task}]",
            f"  train rows={len(tr)}  test rows={len(te)}  index overlap={len(inter)}",
            f"  train estates={len(tr_est)}  test estates={len(te_est)}  "
            f"test estates also in train={len(shared)}  unseen={len(te_est - tr_est)}",
            f"  -> Row-wise split. Reported test metrics are for NEW OBSERVATIONS FROM KNOWN "
            f"ESTATES, not unseen-estate generalisation. See 30_estate_group_cross_validation.",
            "",
        ]
    pd.DataFrame(overlap_rows).to_csv(REPORTS / "30_split_integrity_audit.csv", index=False)

    lines += [
        "Collection-time prediction: REMOVED. No collection-time target, model, report or "
        "artefact is produced anywhere in this project.",
        "",
        "Note: raw workbook data/raw/RubberSense_RawStyle_Synthetic_Dataset.xlsx is never "
        "rewritten by any stage; only copies under data/processed/ and reports/models are generated.",
    ]
    (REPORTS / "30_cv_data_integrity_audit.txt").write_text("\n".join(lines), encoding="utf-8")
    (REPORTS / "30_cv_data_integrity_audit.json").write_text(
        json.dumps({"tasks": overlap_rows}, indent=2, default=str), encoding="utf-8")
    print("\n".join(lines[:8]))
    print("Stage 30 integrity audit complete.")


if __name__ == "__main__":
    main()
