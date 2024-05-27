# run sweep from command line
# python run_sweep.py lgbm True local 3 1

import os
import importlib
import wandb
import warnings
import pandas as pd
import argparse
import shap
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from hyperparams import *
from utils import *

lr = LogisticRegression(
    class_weight="balanced",
    solver="liblinear",
    penalty="l1",
    random_state=2,
    max_iter=420,
)

# Number of random trials
NUM_TRIALS = 3
PROJECT_NAME = "LanternPharma-C2"
ENTITY = None
LABELS = ["Positive", "Negative"]
RANDOM_STATE = 42
warnings.filterwarnings("ignore")
wandb.login()

# create argparser arguments
parser = argparse.ArgumentParser()
# parser.add_argument("model", type=str, default="lgbm")
parser.add_argument("drug", type=str, default=0)
# parser.add_argument("environment", type=str, default="local")
parser.add_argument("splits", type=int, default=4)
parser.add_argument("iters", type=int, default=1)
parser.add_argument("log", type=str, default="no")
# parser.add_argument("usalgo", type=str, default="tomek")
args = parser.parse_args()

label = DRUGS[int(args.drug)]
# print arguments
print("--------------------")
print(
    f"drug: {label}, iters: {args.iters}, cv_splits:{args.splits}, logging: {args.log}"
)
print("--------------------")

# if args.environment == "paperspace":
#     os.chdir("/notebooks/Scripts")

df = pd.read_csv("Data/allTrain.tsv", sep="\t", low_memory=True)
# initialize wandb run
if args.log == "yes":
    run = wandb.init(
        project=PROJECT_NAME,
        entity=ENTITY,
        job_type="modeling",
        save_code=True,
        notes=f"Modelling the lantern pharma drug response dataset with Nested CV",
        tags=[
            f"RandomTrials_{NUM_TRIALS}",
            f"niter{args.iters}",
            f"Drug_{label}",
            f"cv_splits_{args.splits}",
            "2_classes",
            "balanced_accuracy",
        ],
    )

X_train, X_test, y_train, y_test, features, labels = get_train_test_split(df)


def get_model(model_str):
    model_str = model_str.split("'")[1]
    model_ls = model_str.rsplit(".", 1)
    mod = ".".join(model_ls[:-1])
    module = importlib.import_module(mod)
    model_class = getattr(module, model_ls[-1])
    return model_class


def nested_cv(
    pipe,
    X,
    y,
    grid,
    splits,
    iters=5,
    seed=42,
    metrics=["balanced_accuracy", "f1_weighted"],
):
    inner_cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    outer_cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    clf = RandomizedSearchCV(
        estimator=pipe,
        refit=metrics[0],
        n_iter=iters,
        param_distributions=grid,
        cv=inner_cv,
        scoring=metrics,
        n_jobs=-1,
    )
    scores = cross_validate(
        clf, X=X, y=y, cv=outer_cv, scoring=metrics, return_estimator=True
    )
    feat_sels = [
        f'{e.best_estimator_.named_steps["feats"].estimator.__class__}'
        for i, e in enumerate(scores["estimator"])
    ]
    models = [
        f'{e.best_estimator_.named_steps["clf"].__class__}'
        for i, e in enumerate(scores["estimator"])
    ]
    model_params = [
        e.best_estimator_.named_steps["clf"].get_params() for e in scores["estimator"]
    ]
    return {
        "feat_sel": feat_sels,
        "models": models,
        "model_params": model_params,
        "accuracy": scores["test_balanced_accuracy"],
        "f1": scores["test_f1_weighted"],
    }


def run_cvs(pipe, X, y, grid, splits=4, iters=5):
    cv_results = pd.DataFrame()
    row_res = {}

    for i in range(NUM_TRIALS):
        row_res["seed"] = i
        cv_res = nested_cv(pipe, X, y, grid=grid, splits=splits, iters=iters, seed=i)
        row_res.update(cv_res)
        temp_res = pd.DataFrame(row_res, columns=list(row_res.keys()))
        cv_results = pd.concat([cv_results, temp_res], axis=0, ignore_index=True)
        row_res = {}
        print(f"nested cv done for seed {i}")
    return cv_results


def run_test(pipe, exp, X_train, y_train, X_test, y_test):
    pipe.fit(X_train, y_train)
    predictions = pipe.predict(X_test)
    model = pipe.named_steps["clf"]
    feat_bool = pipe.named_steps["feats"].get_support()
    selected_feats = X_train.iloc[:, feat_bool]
    explainer = shap.KernelExplainer(model.predict, selected_feats)
    shap_values = explainer.shap_values(X_test.iloc[:, feat_bool])
    shap.summary_plot(
        shap_values, X_test.iloc[:, feat_bool], show=False, plot_size=(10, 10)
    )
    output_file_path = os.path.join("outputs/", f"{exp}_shap_summary_plot.png")
    plt.savefig(output_file_path)
    plt.close()

    cfr = classification_report(
        y_test, predictions, target_names=LABELS, output_dict=True
    )

    if args.log == "yes":
        # cm = wandb.plot.confusion_matrix(
        #     y_true=y_test, probs=predictions, class_names=LABELS
        # )
        run.log(
            {
                f"shap_values": wandb.Image(data_or_path=output_file_path),
                f"classification_res_cv": cfr,
            }
        )
    else:
        # save clf report
        cfr_db = pd.DataFrame(cfr)
        cfr_db.to_csv(f"outputs/cfr_{exp}.csv")
        # with open(f"outputs/classification_report_{exp}.txt", "w") as f:
        #     f.write(cfr)


# mut_features = features.select_dtypes(include=["int"])
# rna_features = features.select_dtypes(include=["float"])
cv_results = pd.DataFrame()
for model in MODEL_LIST:
    print(f"---------{model}----------")
    pipe = Pipeline([("feats", SelectFromModel(lr)), (f"clf", lr)])
    grid = create_param_grid(model)
    cv_results_ = run_cvs(
        pipe=pipe,
        X=features,
        y=labels[label],
        grid=grid,
        splits=int(args.splits),
        iters=int(args.iters),
    )
    cv_results = pd.concat([cv_results, cv_results_], ignore_index=True)


# # Log model performance
# # eval on custom test set
# i = 0
# for fsm_str, model_str, mp in cv_results[["feat_sel", "models", "model_params"]].values:
#     fsm = get_model(fsm_str)
#     model = get_model(model_str)
#     p = Pipeline(
#         [("feats", SelectFromModel(fsm(), max_features=200)), ("clf", model(**mp))]
#     )
#     run_test(
#         pipe=p,
#         exp=i,
#         X_train=X_train,
#         X_test=X_test,
#         y_train=y_train[label],
#         y_test=y_test[label],
#     )
#     i += 1

cv_results.to_csv(f"outputs/cv_results_{label}.csv", index=False)

if args.log == "yes":
    run.finish()
