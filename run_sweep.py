# run sweep from command line
# python run_sweep.py lgbm True local 3 1

import os
import wandb
import warnings
import pandas as pd
import argparse

from sklearn.preprocessing import (
    MinMaxScaler,
    PolynomialFeatures,
    SplineTransformer,
    KBinsDiscretizer,
    StandardScaler,
    OrdinalEncoder,
)
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as imbPipeline

from sklearn.metrics import (
    get_scorer_names,
    accuracy_score,
    f1_score,
    precision_score,
    confusion_matrix,
    recall_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
)

from hyperparams import *
from util import *

RANDOM_STATE = 42
warnings.filterwarnings("ignore")
wandb.login()

# create argparser arguments
parser = argparse.ArgumentParser()
parser.add_argument("model", type=str, default="lgbm")
parser.add_argument("embedding", type=str, default="no")
parser.add_argument("environment", type=str, default="local")
parser.add_argument("splits", type=int, default=3)
parser.add_argument("iterations", type=int, default=1)
parser.add_argument("log", type=str, default="no")
parser.add_argument("usalgo", type=str, default="tomek")
args = parser.parse_args()

# print arguments
print("--------------------")
print(args.embedding, args.iterations, args.model, args.splits, args.log, args.usalgo)
print("--------------------")

if args.environment == "paperspace":
    os.chdir("/notebooks/Scripts")

# initialize wandb run
if args.log == "yes":
    run = wandb.init(
        project=PROJECT_NAME,
        entity=ENTITY,
        job_type="modeling",
        notes=f"Modelling the ipl2022 dataset with {clfs[args.model]} (5 classes) with feature embeddings={args.embedding}",
        tags=[
            f"niter{args.iterations}",
            f"model{args.model}",
            f"undersample_{args.usalgo}",
            "ipl2022",
            "5_classes",
            "custom_metrics",
        ],
    )

if args.environment == "local":
    if args.embedding == "yes":
        train = pd.read_csv("../Inputs/ball-by-ball prediction/embfeats10K.csv")
    else:
        train = pd.read_csv("../Inputs/ball-by-ball prediction/ipl2022.csv")
else:
    if args.embedding == "yes":
        train = pd.read_csv("embfeats10K.csv")
    else:
        train = pd.read_csv("ipl2022.csv")

if args.embedding == "yes":
    X_train, X_test, y_train, y_test, labels = get_train_test_split(train)
else:
    X_train, X_test, y_train, y_test, labels = get_train_test_split(train)


def calc_metrics(rs, label_dict, training=True, metric="precision"):
    if training:
        y_pred = rs.predict(X_train)
        y_true = y_train
    else:
        y_pred = rs.predict(X_test)
        y_true = y_test

    preds = [label_dict[pred] for pred in y_pred]
    true = [label_dict[pred] for pred in y_true]

    if metric == "precision":
        class_prec = {
            le: ps * 100
            for ps, le in zip(
                precision_score(true, preds, average=None), label_dict.values()
            )
        }
        return class_prec
    elif metric == "recall":
        class_rec = {
            le: ps * 100
            for ps, le in zip(
                recall_score(true, preds, average=None), label_dict.values()
            )
        }
        return class_rec
    else:
        raise ValueError("Metric must be either precision or recall")


cat_features = X_train.select_dtypes(include=["object"]).columns
num_features = X_train.select_dtypes(exclude=["object"]).columns

if args.embedding == "no":
    numeric_transformer = imbPipeline(
        [
            ("poly", PolynomialFeatures(degree=2)),
            ("scaler", StandardScaler()),
            # ("feats", SelectFromModel(lm.Lasso(random_state=RANDOM_STATE))),
        ]
    )
    categorical_transformer = imbPipeline(
        [
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features),
            ("cat", categorical_transformer, cat_features),
        ]
    )
else:
    preprocessor = imbPipeline(
        [
            ("poly", PolynomialFeatures(degree=2)),
            ("scaler", StandardScaler()),
            # ("feats", SelectFromModel(lm.Lasso(random_state=RANDOM_STATE))),
        ]
    )


pipe = imbPipeline(
    [
        ("prep", preprocessor),
        ("sampler", get_sampler(algo=args.usalgo)),
        ("clf", clfs[args.model]),
    ]
)

# --------Custom Scorer---------#
label_dict = {i: labels[i] for i in range(len(labels))}
important_classes = [0, 3, 4]
custom_f1 = get_custom_scorer(important_classes)
# -----------------#

model = clfs[args.model].__class__.__name__
cv = StratifiedKFold(n_splits=args.splits, shuffle=True, random_state=RANDOM_STATE)
rs = RandomizedSearchCV(
    pipe,
    params_wrapper(model=args.model, clf=True),
    n_iter=args.iterations,
    n_jobs=-1,
    cv=cv.split(X_train, y_train),
    scoring=custom_f1,
    random_state=RANDOM_STATE,
)
rs.fit(X_train, y_train)

# Log model performance

predictions = rs.predict(X_test)

if args.log == "yes":
    cm = wandb.plot.confusion_matrix(
        y_true=y_test, preds=predictions, class_names=labels
    )

    wandb.log(
        {
            f"cv_custom_f1_{model}": rs.best_score_,
            f"recall_test_{model}": calc_metrics(
                rs, label_dict, training=False, metric="recall"
            ),
            f"precision_test_{model}": calc_metrics(
                rs, label_dict, training=False, metric="precision"
            ),
            "best_params": rs.best_params_,
            "conf_mat": cm,
        }
    )

    run.finish()
else:
    print(
        f"cv_custom_f1_{model}: {rs.best_score_}",
        f"recall_test_{model}: {calc_metrics(rs, label_dict, training=False, metric='recall')}",
        f"precision_test_{model}: {calc_metrics(rs, label_dict, training=False, metric='precision')}",
        f"best_params_{args.model}: {rs.best_params_}",
        sep="\n",
    )
