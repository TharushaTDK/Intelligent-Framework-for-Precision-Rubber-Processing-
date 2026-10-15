"""Baseline regression candidates and shared configuration."""
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
def get_candidates():
    return {
      "LinearRegression": LinearRegression(),
      "Ridge": Ridge(alpha=1.0),
      "RandomForest": RandomForestRegressor(n_estimators=200,min_samples_leaf=2,random_state=42,n_jobs=-1),
      "ExtraTrees": ExtraTreesRegressor(n_estimators=200,min_samples_leaf=2,random_state=42,n_jobs=-1),
      "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
    }
