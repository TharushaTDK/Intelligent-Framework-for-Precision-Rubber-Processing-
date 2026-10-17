from stage_utils import *
df=load_data()
rows=[]
for col, lo, hi in [('latex_drc_pct',0,100),('price_lkr_per_kg',0,10000)]:
    s=pd.to_numeric(df[col],errors='coerce')
    rows.append({'target':col,'n':int(s.notna().sum()),'missing':int(s.isna().sum()),'min':float(s.min()) if s.notna().any() else None,'max':float(s.max()) if s.notna().any() else None,'outside_expected_range':int((~s.between(lo,hi)&s.notna()).sum()),'target_type':'regression'})
rows.append({'target':'tapping_suitability_proxy','n':int(df.shape[0]),'missing':int(df['record_id'].isna().sum()),'min':None,'max':None,'outside_expected_range':0,'target_type':'classification proxy; rule-derived with simulated uncertainty'})
pd.DataFrame(rows).to_csv(REPORTS/'02_target_validation.csv',index=False)
(REPORTS/'02_target_validation.txt').write_text('Target validation completed. Tapping target is a synthetic proxy, not expert-validated advice.\n',encoding='utf-8')
print(pd.DataFrame(rows).to_string(index=False))
