"""Optional XGBoost candidate; requires xgboost to be installed."""
def build_model():
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, objective="reg:squarederror", random_state=42, n_jobs=-1)
