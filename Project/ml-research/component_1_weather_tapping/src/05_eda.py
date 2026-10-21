from stage_utils import *
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:
    plt = None
FIGDIR = REPORTS.parent / 'figures' / '05_eda'
df=load_data()
num=df.select_dtypes(include='number')
num.describe().T.to_csv(REPORTS/'05_numeric_descriptive_statistics.csv')
df.nunique(dropna=False).rename('unique_values').to_csv(REPORTS/'05_column_cardinality.csv')
df['visible_latex_condition'].value_counts(dropna=False).rename_axis('visible_latex_condition').reset_index(name='records').to_csv(REPORTS/'05_records_by_latex_condition.csv',index=False)
df['estate_id'].value_counts(dropna=False).rename_axis('estate_id').reset_index(name='records').to_csv(REPORTS/'05_records_by_estate.csv',index=False)
if plt is not None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for col in ['latex_drc_pct','price_lkr_per_kg','temperature_c','soil_moisture_pct','soil_ph']:
        s=pd.to_numeric(df[col],errors='coerce').dropna()
        if len(s):
            plt.figure(figsize=(7,4)); plt.hist(s,bins=24); plt.title(f'Distribution: {col}'); plt.xlabel(col); plt.ylabel('Records'); plt.tight_layout(); plt.savefig(FIGDIR/f'{col}_distribution.png',dpi=140); plt.close()
    print('Stage 05 EDA reports and figures saved.')
else:
    print('Stage 05 EDA reports saved; matplotlib not installed so figures were skipped.')
