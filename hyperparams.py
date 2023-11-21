import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, Union

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    BaggingClassifier,
)

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from lightgbm.sklearn import LGBMClassifier
from xgboost.sklearn import XGBClassifier

import sklearn.linear_model as lm
from sklearn.metrics import f1_score

RANDOM_STATE = 42
MODEL_LIST = [
    "lr",
    "rf",
    "gbr",
    "bag",
    "svc",
    "lgbm",
    "knn",
    "ada",
    "et",
    "xgb",
]
FSM_LIST = ["lr", "rf", "lgbm", "xgb"]


class NaiveClassifier(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        self.classes_, self.counts_ = np.unique(y, return_counts=True)
        self.prior_ = self.counts_ / len(y)
        return self

    def predict(self, X):
        return np.random.choice(self.classes_, size=len(X), p=self.prior_)

    def score(self, X, y):
        return f1_score(y, self.predict(X))

    def __str__(self) -> str:
        return "NaiveClassifier"


clfs = {
    "naive": NaiveClassifier(),
    "lr": lm.LogisticRegression(
        random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1, max_iter=500
    ),
    "rf": RandomForestClassifier(
        random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1
    ),
    "gbr": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "bag": BaggingClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    "svc": SVC(random_state=RANDOM_STATE, class_weight="balanced", probability=True),
    "lgbm": LGBMClassifier(
        max_depth=5,
        num_leaves=3,
        class_weight="balanced",
        min_child_samples=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    ),
    "knn": KNeighborsClassifier(n_jobs=-1),
    "ada": AdaBoostClassifier(random_state=RANDOM_STATE),
    "et": ExtraTreesClassifier(
        random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1
    ),
    "xgb": XGBClassifier(
        booster="gbtree", tree_method="hist", random_state=RANDOM_STATE
    ),
    "dtc": DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
}


def get_train_test_split(df):
    # constant_columns = [col for col in df.columns if df[col].nunique() == 1]
    df_ = df.copy()
    df_.type = pd.Categorical(df_.type.astype("category"))
    df_.type = df_.type.cat.codes
    test_idx = [2, 4, 5, 13, 15, 19, 24, 30, 34, 36, 38, 40, 44, 46]
    labels = df_[
        [
            "Rapamycin_response",
            "Mitomycin_response",
            "Fulvestrant_response",
            "Gefitinib_response",
            "Rapamycin-Gefitinib_response",
            "Mitomycin-Fulvestrant_response",
        ]
    ]
    features = df_.drop(columns=labels.columns.to_list(), axis=1)
    X_train, X_test, y_train, y_test = (
        features.drop(test_idx, axis=0).reset_index(drop=True),
        features.iloc[features.index[test_idx]],
        labels.drop(test_idx, axis=0).reset_index(drop=True),
        labels.iloc[labels.index[test_idx]],
    )
    return X_train, X_test, y_train, y_test, features, labels


lr_params = {
    "fit_intercept": [True, False],
    "C": np.linspace(0.01, 1.0, 10, dtype=np.float16),
    "l1_ratio": np.linspace(0.0, 1.0, 10, dtype=np.float32),
    "solver": ["saga"],
    "penalty": ["elasticnet"],
    "max_iter": [200, 300, 400],
}

rf_params = {
    "n_estimators": np.linspace(50, 150, 20, dtype=np.int16),
    "max_depth": np.linspace(2, 20, 10, dtype=np.int16),
    "min_samples_split": np.linspace(2, 10, 5, dtype=np.int16),
    "min_samples_leaf": np.linspace(2, 10, 5, dtype=np.int16),
    "max_features": np.linspace(0.2, 1, 20, dtype=np.float16),
    # 'clf__learning_rate': np.linspace(0.01, 1, 50, dtype=np.float16),
    "criterion": ["gini", "entropy", "log_loss"],
    # 'clf__bootstrap': [True, False],
    # 'clf__loss': ['log_loss', 'exponential'],
    "max_samples": np.linspace(0.2, 1.0, 10, dtype=np.float16),
    "ccp_alpha": np.linspace(0.0, 5.0, 20, dtype=np.float16),
    # "warm_start": [True, False],
    # 'clf__n_iter_no_change': np.linspace(1, 10, 10, dtype=np.int16),
    # 'clf__min_impurity_decrease': np.linspace(0.0001, 10.0, 10, dtype=np.float16),
}

gbr_params = {
    "n_estimators": np.linspace(50, 150, 20, dtype=np.int16),
    "max_depth": np.linspace(2, 20, 10, dtype=np.int16),
    "min_samples_split": np.linspace(2, 10, 5, dtype=np.int16),
    "min_samples_leaf": np.linspace(2, 10, 5, dtype=np.int16),
    "max_features": np.linspace(0.2, 1, 20, dtype=np.float16),
    "learning_rate": np.linspace(1e-3, 1.0, 10, dtype=np.float16),
    "criterion": ["friedman_mse", "squared_error"],
    # 'clf__bootstrap': [True, False],
    "loss": ["log_loss"],
    "subsample": np.linspace(0.2, 1.0, 10, dtype=np.float16),
    # 'clf__ccp_alpha': np.linspace(0.0, 5.0, 10, dtype=np.float16),
    "warm_start": [True, False],
    # "n_iter_no_change": np.linspace(2, 50, 20, dtype=np.int16),
    # 'clf__min_impurity_decrease': np.linspace(0.0, 10.0, 20, dtype=np.float16),
}

bag_params = {
    "estimator": [clfs["knn"]],
    "n_estimators": np.linspace(20, 100, 10, dtype=np.int16),
    "max_samples": np.linspace(0.2, 1.0, 10, dtype=np.float16),
    "bootstrap": [True, False],
    "warm_start": [True, False],
}

svc_params = {
    "C": np.linspace(0.01, 2.0, 20, dtype=np.float16),
    "kernel": ["linear", "poly", "rbf"],
    "degree": np.linspace(1, 4, 3, dtype=np.int16),
    "gamma": np.linspace(0.001, 2.0, 20, dtype=np.float16),
    "shrinking": [True, False],
}

knn_params = {
    "n_neighbors": np.linspace(3, 6, 10, dtype=np.int16),
    "weights": ["uniform", "distance"],
    "algorithm": ["ball_tree", "kd_tree", "brute"],
    "leaf_size": np.linspace(2, 7, 5, dtype=np.int16),
    # "p": np.linspace(1, 3, 3, dtype=np.int16),
    # "metric": ["euclidean", "manhattan", "chebyshev", "minkowski"],
}

lgbm_params = {
    "n_estimators": np.linspace(50, 100, 10, dtype=np.int16),
    "max_depth": np.linspace(2, 20, 20, dtype=np.int16),
    "num_leaves": np.linspace(2, 5, 3, dtype=np.int16),
    "learning_rate": np.linspace(1e-3, 1.0, 40, dtype=np.float16),
    "subsample": np.linspace(0.2, 1.0, 40, dtype=np.float16),
    "colsample_bytree": np.linspace(0.2, 1.0, 40, dtype=np.float16),
    "reg_alpha": np.linspace(0.0, 1.0, 20, dtype=np.float16),
    "reg_lambda": np.linspace(0.0, 1.0, 20, dtype=np.float16),
    "min_child_samples": np.linspace(2, 7, 5, dtype=np.int16),
}

xgb_params = {
    "n_estimators": np.linspace(50, 100, 10, dtype=np.int16),
    "max_depth": np.linspace(2, 20, 10, dtype=np.int16),
    "min_child_weight": np.linspace(0.001, 1.0, 40, dtype=np.float16),
    "subsample": np.linspace(0.2, 1.0, 40, dtype=np.float16),
    "colsample_bytree": np.linspace(0.1, 1.0, 40, dtype=np.float16),
    "colsample_bylevel": np.linspace(0.1, 1.0, 40, dtype=np.float16),
    "colsample_bynode": np.linspace(0.1, 1.0, 40, dtype=np.float16),
    # "num_round": np.linspace(10, 50, 10, dtype=np.int16),
    "learning_rate": np.linspace(0.1, 0.4, 20, dtype=np.float16),
    "reg_alpha": np.linspace(0.001, 1.0, 20, dtype=np.float16),
    "reg_lambda": np.linspace(0.001, 1.0, 20, dtype=np.float16),
    "gamma": np.linspace(0.1, 0.8, 20, dtype=np.float16),
    # "scale_pos_weight": np.linspace(1, 100, 25, dtype=np.int16),
    "monotone_constraints": [None, (1, -1), (-1, 1)],
    # "interaction_constraints": [None, "[]", "[[0, 1]]"],
    # "max_bin": np.linspace(400, 600, 10, dtype=np.int16),
    "grow_policy": ["depthwise", "lossguide"],
    # "max_leaves": np.linspace(0, 100, 10, dtype=np.int16),
}

spline_params = {
    "splines__degree": np.linspace(1, 3, 3, dtype=np.int16),
    "splines__n_knots": np.linspace(2, 10, 10, dtype=np.int16),
    # "splines__include_bias": [True, False],
    # "splines__strategy": ["quantile", "uniform"],
}

kbins_params = {
    "bins__n_bins": np.linspace(2, 10, 5, dtype=np.int16),
    # "encode": ["ordinal", "onehot-dense"],
    "bins__strategy": ["uniform", "quantile", "kmeans"],
}

poly_params = {
    "poly__degree": np.linspace(1, 3, 3, dtype=np.int16),
    "poly__interaction_only": [True],
    "poly__include_bias": [True, False],
}

fsm_params = {
    "feats__estimator": [clfs[fsm] for fsm in FSM_LIST],
}


# a function to append 'model__' to the beginning of each parameter name
def appendprix(params, prix="clf__"):
    new_params = {}
    for key, value in params.items():
        if key == "clf":
            new_params[key] = value
        else:
            new_params[prix + key] = value
    return new_params


# get the parameters for each model
def gethps_(model, clf=True):
    if model == "rf":
        rf_params.update({"clf": [clfs[model]]})
        if clf:
            return appendprix(rf_params)
        else:
            return rf_params
    elif model == "bag":
        bag_params.update({"clf": [clfs[model]]})
        if clf:
            return appendprix(bag_params)
        else:
            return bag_params
    elif model == "gbr":
        gbr_params.update({"clf": [clfs[model]]})
        if clf:
            return appendprix(gbr_params)
        else:
            return gbr_params
    elif model == "svc":
        svc_params.update({"clf": [clfs[model]]})
        if clf:
            return appendprix(svc_params)
        else:
            return svc_params
    elif model == "lgbm":
        lgbm_params.update({"clf": [clfs[model]]})
        if clf:
            return appendprix(lgbm_params)
        else:
            return lgbm_params
    elif model == "xgb":
        xgb_params.update({"clf": [clfs[model]]})
        if clf:
            return appendprix(xgb_params)
        else:
            return xgb_params
    elif model == "lr":
        lr_params.update({"clf": [clfs[model]]})
        if clf:
            return appendprix(lr_params)
        else:
            return lr_params
    elif model == "knn":
        knn_params.update({"clf": [clfs[model]]})
        if clf:
            return appendprix(knn_params)
        else:
            return knn_params
    else:
        return {"clf": [clfs[model]]}


def create_param_grid(model):
    grid = {}
    grid.update(fsm_params)
    params = gethps_(model, clf=True)
    grid.update(params)
    return grid
