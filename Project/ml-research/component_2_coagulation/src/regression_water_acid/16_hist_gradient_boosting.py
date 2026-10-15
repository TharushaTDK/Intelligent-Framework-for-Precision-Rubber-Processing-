"""Histogram Gradient Boosting candidate reference."""
from sklearn.ensemble import HistGradientBoostingRegressor

def build_model():
    return HistGradientBoostingRegressor(random_state=42)
