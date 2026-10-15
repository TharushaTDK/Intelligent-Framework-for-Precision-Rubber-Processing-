"""Gradient Boosting candidate reference."""
from sklearn.ensemble import GradientBoostingRegressor

def build_model():
    return GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=2, random_state=42)
