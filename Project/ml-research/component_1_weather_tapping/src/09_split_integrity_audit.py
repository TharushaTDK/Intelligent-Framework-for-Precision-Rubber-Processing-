from stage_utils import *
splitdir=ROOT/'data/splits'; rows=[]
for p in sorted(splitdir.glob('*indices.csv')):
    try:
        x=pd.read_csv(p); col='row_index' if 'row_index' in x else x.columns[0]
        rows.append({'file':p.name,'n_rows':len(x),'unique_indices':int(x[col].nunique()),'duplicate_indices':int(x[col].duplicated().sum()),'min_index':int(x[col].min()) if len(x) else None,'max_index':int(x[col].max()) if len(x) else None})
    except Exception as e: rows.append({'file':p.name,'error':str(e)})
pd.DataFrame(rows).to_csv(REPORTS/'09_split_integrity_audit.csv',index=False)
(REPORTS/'09_split_integrity_audit.txt').write_text('Audit reflects existing row-based splits. For generalization to entirely unseen estates, use group-based splitting by estate_id and report that separately.\n',encoding='utf-8')
print(pd.DataFrame(rows).to_string(index=False))
