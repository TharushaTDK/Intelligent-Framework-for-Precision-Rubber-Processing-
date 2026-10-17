from stage_utils import *
df=load_data(); ts=pd.to_datetime(df['timestamp_local'],errors='coerce'); df['month']=ts.dt.month; df['hour']=ts.dt.hour
for col in ['latex_drc_pct','price_lkr_per_kg','temperature_c','relative_humidity_pct','rainfall_last_6h_mm']:
    pd.to_numeric(df[col],errors='coerce').groupby(df['month']).agg(['count','mean','std','min','max']).to_csv(REPORTS/f'06_monthly_{col}_summary.csv')
summary=df.groupby('estate_id',dropna=False).agg(records=('record_id','count'),mean_drc=('latex_drc_pct','mean'),median_drc=('latex_drc_pct','median'),std_drc=('latex_drc_pct','std'),mean_price=('price_lkr_per_kg','mean')).reset_index()
summary.to_csv(REPORTS/'06_estate_profile.csv',index=False)
print('Stage 06 temporal and estate summaries saved.')
