"""Random Forest candidate reference."""
from sklearn.ensemble import RandomForestRegressor

def build_model():
    return RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1)
