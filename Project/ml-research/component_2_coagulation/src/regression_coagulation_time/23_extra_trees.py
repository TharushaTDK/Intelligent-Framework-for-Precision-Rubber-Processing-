"""Extra Trees candidate reference."""
from sklearn.ensemble import ExtraTreesRegressor

def build_model():
    return ExtraTreesRegressor(n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1)
