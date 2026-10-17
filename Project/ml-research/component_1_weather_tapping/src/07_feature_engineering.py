import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import load_raw, clean_basic, PROCESSED, ensure_dirs
ensure_dirs(); df=clean_basic(load_raw())
# Safe, available derived features; timestamps are not target leakage for a collection-time task (that task is removed).
ts=__import__('pandas').to_datetime(df['timestamp_local'],errors='coerce')
df['month_sin']=__import__('numpy').sin(2*__import__('numpy').pi*ts.dt.month.fillna(1)/12)
df['month_cos']=__import__('numpy').cos(2*__import__('numpy').pi*ts.dt.month.fillna(1)/12)
df['weather_rain_risk']=(df['rainfall_last_6h_mm'].fillna(0)*0.5+df['forecast_rain_next_6h_mm'].fillna(0)*0.5)
df.to_csv(PROCESSED/'feature_engineered_dataset.csv',index=False)
print(f'Saved {len(df)} rows to {PROCESSED / "feature_engineered_dataset.csv"}')
