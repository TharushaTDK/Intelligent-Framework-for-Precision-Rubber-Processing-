from common import ensure_dirs, load_raw, REPORTS
def main():
    ensure_dirs()
    df = load_raw()
    report = [
        "RAW DATA QUALITY REPORT",
        f"Shape: {df.shape}",
        f"Total missing cells: {int(df.isna().sum().sum())}",
        "",
        "Missing values by column:",
        df.isna().sum().to_string(),
        "",
        f"Exact duplicate rows (excluding record_id): {int(df.drop(columns=['record_id'], errors='ignore').duplicated().sum())}",
        f"Duplicate record IDs: {int(df['record_id'].duplicated().sum())}",
        f"Negative rainfall values: {int((df['rainfall_last_6h_mm'] < 0).sum())}",
        f"Soil pH outside 0-14: {int((~df['soil_ph'].between(0,14)).sum())}",
        "",
        "Review source provenance before using any row as research evidence."
    ]
    (REPORTS / "data_quality_report.txt").write_text("\\n".join(report), encoding="utf-8")
    print("\\n".join(report[:6]))
if __name__ == "__main__": main()
