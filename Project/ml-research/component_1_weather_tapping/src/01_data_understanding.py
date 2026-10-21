from common import ensure_dirs, load_raw, REPORTS, DATASET_PROVENANCE
def main():
    ensure_dirs()
    df = load_raw()
    lines = [
        "DATASET OVERVIEW — RubberSense Component 1",
        f"Rows: {len(df)}",
        f"Columns: {len(df.columns)}",
        "",
        "Columns:",
        *[f"- {c} ({df[c].dtype})" for c in df.columns],
        "",
        f"Dataset provenance: {DATASET_PROVENANCE}.",
        "Expected target columns: latex_drc_pct, tapping_suitability_proxy, price_lkr_per_kg. Collection-time prediction is excluded because no suitable collection-time training data is available. Tapping labels are a rainfall-rule proxy, not expert advice."
    ]
    (REPORTS / "01_dataset_overview.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:5]))
if __name__ == "__main__": main()
