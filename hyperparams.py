import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, Union

from imblearn.under_sampling import (
    NearMiss,
    CondensedNearestNeighbour,
    TomekLinks,
    RandomUnderSampler,
    EditedNearestNeighbours,
    RepeatedEditedNearestNeighbours,
    AllKNN,
    InstanceHardnessThreshold,
    NeighbourhoodCleaningRule,
)

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    BaggingClassifier,
)

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

import sklearn.linear_model as lm
from sklearn.metrics import f1_score

RANDOM_STATE = 42


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


clfs = {
    "naive": NaiveClassifier(),
    "lr": lm.LogisticRegression(random_state=RANDOM_STATE, n_jobs=-1),
    "rf": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    "gbr": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "bag": BaggingClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    "svc": SVC(random_state=RANDOM_STATE),
    "lgbm": LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    "knn": KNeighborsClassifier(n_jobs=-1),
    "ada": AdaBoostClassifier(random_state=RANDOM_STATE),
    "et": ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    "xgb": XGBClassifier(
        booster="gbtree", tree_method="hist", random_state=RANDOM_STATE
    ),
}


def get_sampler(algo="tomek"):
    if algo == "tomek":
        return TomekLinks(n_jobs=-1)
    elif algo == "nm":
        return NearMiss(version=3, n_neighbors=3, n_jobs=-1)
    elif algo == "cnn":
        return CondensedNearestNeighbour(
            n_neighbors=3, random_state=RANDOM_STATE, n_jobs=-1
        )
    elif algo == "rus":
        return RandomUnderSampler(random_state=RANDOM_STATE)
    elif algo == "enn":
        return EditedNearestNeighbours(n_neighbors=3, kind_sel="mode", n_jobs=-1)
    elif algo == "renn":
        return RepeatedEditedNearestNeighbours(
            n_neighbors=3, kind_sel="mode", max_iter=100, n_jobs=-1
        )
    elif algo == "allknn":
        return AllKNN(n_neighbors=3, kind_sel="mode", allow_minority=False, n_jobs=-1)
    elif algo == "iht":
        return InstanceHardnessThreshold(n_jobs=-1)
    elif algo == "ncr":
        return NeighbourhoodCleaningRule(
            n_neighbors=3, kind_sel="mode", threshold_cleaning=0.5, n_jobs=-1
        )
    else:
        return None


def get_train_test_split(df):
    constant_columns = [col for col in df.columns if df[col].nunique() == 1]
    df.type = pd.Categorical(df.type.astype("category"))
    df.type = df.type.cat.codes
    test_idx = [2, 4, 5, 13, 15, 19, 24, 30, 34, 36, 38, 40, 44, 46]
    labels = df[
        [
            "Rapamycin_response",
            "Mitomycin_response",
            "Fulvestrant_response",
            "Gefitinib_response",
            "Rapamycin-Gefitinib_response",
            "Mitomycin-Fulvestrant_response",
        ]
    ]
    features = df.drop(columns=labels.columns.to_list() + constant_columns, axis=1)
    X_train, X_test, y_train, y_test = (
        features.drop(test_idx, axis=0).reset_index(drop=True),
        features.iloc[features.index[test_idx]],
        labels.drop(test_idx, axis=0).reset_index(drop=True),
        labels.iloc[labels.index[test_idx]],
    )
    return X_train, X_test, y_train, y_test


# Split ipl2022 data into (pp,m,d) or power play (1-6), middle overs (7-15), death (15-20) to model them separately
def split_match_phases(X, y):
    df = pd.concat([X, y], axis=1)
    # split data into power play, middle overs and death overs
    df_pp = df[df["overs"].isin([1, 2, 3, 4, 5, 6])]
    df_mo = df[df["overs"].isin([7, 8, 9, 10, 11, 12, 13, 14, 15])]
    df_d = df[df["overs"].isin([16, 17, 18, 19, 20])]

    return df_pp, df_mo, df_d


def get_match_phases(df, phase="pp"):
    X_train, X_test, y_train, y_test, labels = get_train_test_split(df)
    df_train_ = pd.concat([X_train, pd.DataFrame(y_train, columns=["target"])], axis=1)
    df_test_ = pd.concat([X_test, pd.DataFrame(y_test, columns=["target"])], axis=1)
    df_train, df_test = split_match_phases(df_train_, phase), split_match_phases(
        df_test_, phase
    )
    X_train, y_train = df_train.drop("target", axis=1), df_train["target"]
    X_test, y_test = df_test.drop("target", axis=1), df_test["target"]
    return X_train, X_test, y_train, y_test, labels


lr_params = {
    "fit_intercept": [True, False],
    "C": np.linspace(0.5, 1.0, 10, dtype=np.float32),
    "l1_ratio": np.linspace(0.5, 1.0, 10, dtype=np.float32),
    "solver": ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
    "penalty": ["elasticnet"],
}

rf_params = {
    "n_estimators": np.linspace(100, 800, 20, dtype=np.int16),
    "max_depth": np.linspace(2, 100, 40, dtype=np.int16),
    "min_samples_split": np.linspace(2, 40, 20, dtype=np.int16),
    "min_samples_leaf": np.linspace(2, 40, 20, dtype=np.int16),
    "max_features": np.linspace(0.2, 1, 20, dtype=np.float16),
    # 'clf__learning_rate': np.linspace(0.01, 1, 50, dtype=np.float16),
    "criterion": ["gini", "entropy", "log_loss"],
    # 'clf__bootstrap': [True, False],
    # 'clf__loss': ['log_loss', 'exponential'],
    "max_samples": np.linspace(0.2, 1.0, 10, dtype=np.float16),
    "ccp_alpha": np.linspace(0.0, 5.0, 20, dtype=np.float16),
    "warm_start": [True, False],
    # 'clf__n_iter_no_change': np.linspace(1, 10, 10, dtype=np.int16),
    # 'clf__min_impurity_decrease': np.linspace(0.0001, 10.0, 10, dtype=np.float16),
}

gbr_params = {
    "n_estimators": np.linspace(100, 500, 20, dtype=np.int16),
    "max_depth": np.linspace(2, 100, 40, dtype=np.int16),
    "min_samples_split": np.linspace(2, 40, 20, dtype=np.int16),
    "min_samples_leaf": np.linspace(2, 40, 20, dtype=np.int16),
    "max_features": np.linspace(0.2, 1, 20, dtype=np.float16),
    "learning_rate": np.linspace(1e-3, 1.0, 40, dtype=np.float16),
    "criterion": ["friedman_mse", "squared_error"],
    # 'clf__bootstrap': [True, False],
    "loss": ["log_loss"],
    "subsample": np.linspace(0.2, 1.0, 40, dtype=np.float16),
    # 'clf__ccp_alpha': np.linspace(0.0, 5.0, 10, dtype=np.float16),
    "warm_start": [True, False],
    "n_iter_no_change": np.linspace(2, 50, 20, dtype=np.int16),
    # 'clf__min_impurity_decrease': np.linspace(0.0, 10.0, 20, dtype=np.float16),
}

bag_params = {
    # 'estimator': [GaussianNB(), DecisionTreeClassifier(random_state=42)],
    "n_estimators": np.linspace(20, 500, 20, dtype=np.int16),
    "max_samples": np.linspace(0.2, 1.0, 10, dtype=np.float16),
    "bootstrap": [True, False],
    "warm_start": [True, False],
}

svc_params = {
    "C": np.linspace(0.1, 10.0, 40, dtype=np.float16),
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
    "degree": np.linspace(1, 20, 10, dtype=np.int16),
    "gamma": np.linspace(0.0001, 10.0, 40, dtype=np.float16),
    "shrinking": [True, False],
    "tol": np.linspace(0.00001, 0.001, 20, dtype=np.float16),
    "max_ter": np.linspace(100, 1000, 20, dtype=np.int16),
}

knn_params = {
    "n_neighbors": np.linspace(3, 20, 10, dtype=np.int16),
    "weights": ["uniform", "distance"],
    "algorithm": ["ball_tree", "kd_tree", "brute"],
    "leaf_size": np.linspace(10, 100, 40, dtype=np.int16),
    # "p": np.linspace(1, 3, 3, dtype=np.int16),
    # "metric": ["euclidean", "manhattan", "chebyshev", "minkowski"],
}

lgbm_params = {
    "objective": ["multiclass", "multiclassova"],
    "n_estimators": np.linspace(100, 500, 10, dtype=np.int16),
    "max_depth": np.linspace(2, 100, 40, dtype=np.int16),
    "num_leaves": np.linspace(2, 80, 40, dtype=np.int16),
    "learning_rate": np.linspace(1e-3, 1.0, 40, dtype=np.float16),
    "bagging_fraction": np.linspace(0.2, 1.0, 40, dtype=np.float16),
    "bagging_freq": np.linspace(0, 10, 10, dtype=np.int16),
    "colsample_bytree": np.linspace(0.2, 1.0, 40, dtype=np.float16),
    # "colsample_bynode": np.linspace(0.1, 1.0, 40, dtype=np.float16),
    "reg_alpha": np.linspace(0.0, 1.0, 20, dtype=np.float16),
    "reg_lambda": np.linspace(0.0, 1.0, 20, dtype=np.float16),
    # "min_child_samples": np.linspace(2, 100, 40, dtype=np.int16),
    "min_data_in_leaf": np.linspace(100, 2000, 40, dtype=np.int16),
    "min_child_weight": np.linspace(0.0001, 0.1, 20, dtype=np.float16),
    "cat_l2": np.linspace(1.0, 20.0, 10, dtype=np.float16),
}

xgb_params = {
    "n_estimators": np.linspace(100, 500, 10, dtype=np.int16),
    # "max_depth": np.linspace(10, 100, 40, dtype=np.int16),
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

sfm_params = {
    "feats__threshold": ["mean", "median"],
}


# a function to append 'clf__' to the beginning of each parameter name
def appendprix(params, prix="clf__"):
    new_params = {}
    for key, value in params.items():
        new_params[prix + key] = value
    return new_params


# get the parameters for each model
def gethps_(model, clf=True):
    if model == "rf":
        if clf:
            return appendprix(rf_params)
        else:
            return rf_params
    elif model == "bag":
        if clf:
            return appendprix(bag_params)
        else:
            return bag_params
    elif model == "gbr":
        if clf:
            return appendprix(gbr_params)
        else:
            return gbr_params
    elif model == "svc":
        if clf:
            return appendprix(svc_params)
        else:
            return svc_params
    elif model == "lgbm":
        if clf:
            return appendprix(lgbm_params)
        else:
            return lgbm_params
    elif model == "xgb":
        if clf:
            return appendprix(xgb_params)
        else:
            return xgb_params
    elif model == "lr":
        if clf:
            return appendprix(lr_params)
        else:
            return lr_params
    elif model == "knn":
        if clf:
            return appendprix(knn_params)
        else:
            return knn_params
    else:
        raise ValueError("Model not found")


def params_wrapper(model, clf=True):
    new_params = gethps_(model, clf)
    # prep_params = dict(sfm_params, **poly_params)
    prep_params = poly_params.copy()
    # prep_params = dict(prep_params, **sfm_params)
    # prep_params = dict(prep_params, **spline_params)
    new_params.update(appendprix(prep_params, prix="prep__num__"))
    return new_params
