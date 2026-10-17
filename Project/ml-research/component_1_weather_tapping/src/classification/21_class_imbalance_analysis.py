from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import ROOT, load_raw, clean_basic, REPORTS
import pandas as pd
x=clean_basic(load_raw())['tapping_suitability_proxy'].astype(str).value_counts(dropna=False).rename_axis('class').reset_index(name='count')
x['proportion']=x['count']/x['count'].sum(); x.to_csv(REPORTS/'21_tapping_class_distribution.csv',index=False)
print(x.to_string(index=False))
