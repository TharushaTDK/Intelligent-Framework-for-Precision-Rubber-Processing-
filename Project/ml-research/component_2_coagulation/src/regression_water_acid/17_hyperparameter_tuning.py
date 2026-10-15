"""Small GridSearchCV tuning helper; final tuning should use training data only."""
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
def tune_random_forest(X_train,y_train):
    pipe=Pipeline([("imputer",SimpleImputer(strategy="median")),
                   ("model",RandomForestRegressor(random_state=42,n_jobs=-1))])
    grid={"model__n_estimators":[100,200],"model__max_depth":[None,10],"model__min_samples_leaf":[1,2]}
    search=GridSearchCV(pipe,grid,cv=5,scoring="neg_mean_absolute_error",n_jobs=-1)
    search.fit(X_train,y_train)
    return search.best_estimator_,search.best_params_,-search.best_score_
