from common import ensure_dirs, load_raw, clean_basic, PROCESSED, REPORTS
def main():
    ensure_dirs()
    raw = load_raw()
    clean = clean_basic(raw)
    clean.to_csv(PROCESSED / "preprocessed_raw_style.csv", index=False)
    (REPORTS / "03_preprocessing_report.txt").write_text(
        "Preprocessing applied to a copy of the prepared farmer dataset (stage 00):\n"
        "- invalid negative rainfall set to missing\n"
        "- plausible physical bounds checked for pH, humidity, soil moisture, temperature, DRC\n"
        "- categorical whitespace/case normalized\n"
        "- month/day-of-year features extracted\n"
        "- rainfall-rule tapping proxy target derived (no added noise)\n"
        "- missing values are imputed inside model pipelines to avoid train/test leakage\n"
        "- raw workbook remains unchanged\n"
        "Warning: DRC values were normalized by the dataset author and tapping labels are rule-derived;\n"
        "neither is verified field ground truth.\n",
        encoding="utf-8")
    print("Saved", PROCESSED / "preprocessed_raw_style.csv")
if __name__ == "__main__": main()
