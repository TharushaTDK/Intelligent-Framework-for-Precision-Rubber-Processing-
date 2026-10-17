from stage_utils import *
df=load_data(); num=df.select_dtypes(include='number')
num.corr(numeric_only=True).to_csv(REPORTS/'08_numeric_correlation_matrix.csv')
for target in ['latex_drc_pct','price_lkr_per_kg']:
    if target in num:
        num.corr(numeric_only=True)[target].drop(target).sort_values(key=lambda s:s.abs(),ascending=False).rename('correlation_with_target').to_csv(REPORTS/f'08_{target}_correlations.csv')
(REPORTS/'08_feature_selection_notes.txt').write_text('Feature correlations are exploratory and do not prove causation. Estate ID is categorical and may not generalize to unseen estates. Collection-time prediction is excluded because no suitable training target is available.\n',encoding='utf-8')
print('Stage 08 feature analysis saved.')
