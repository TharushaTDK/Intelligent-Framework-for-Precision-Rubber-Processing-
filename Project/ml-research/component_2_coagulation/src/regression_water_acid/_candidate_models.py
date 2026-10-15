"""Five regression candidates; XGBoost replaces GradientBoosting if installed."""
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
def candidates():
    models={
      "LinearRegression":LinearRegression(),
      "Ridge":Ridge(alpha=1.0),
      "RandomForest":RandomForestRegressor(n_estimators=200,min_samples_leaf=2,random_state=42,n_jobs=-1),
      "ExtraTrees":ExtraTreesRegressor(n_estimators=200,min_samples_leaf=2,random_state=42,n_jobs=-1),
      "HistGradientBoosting":HistGradientBoostingRegressor(random_state=42),
    }
    try:
        from xgboost import XGBRegressor
        models.pop("HistGradientBoosting")
        models["XGBoost"]=XGBRegressor(n_estimators=200,max_depth=4,learning_rate=0.05,objective="reg:squarederror",random_state=42,n_jobs=-1)
    except ImportError:
        pass
    return models
