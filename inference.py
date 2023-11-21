import argparse
import pandas as pd
import numpy as np
import ast
import pickle

from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from hyperparams import get_train_test_split
from utils import *

# ful_res = pd.read_csv("outputs/cv_results_Fulvestrant_response.csv")
# gef_res = pd.read_csv("outputs/cv_results_Gefitinib_response.csv")
# mit_res = pd.read_csv("outputs/cv_results_Mitomycin_response.csv")
# rap_res = pd.read_csv("outputs/cv_results_Rapamycin_response.csv")
# mitful_res = pd.read_csv("outputs/cv_results_Mitomycin-Fulvestrant_response.csv")
# rapgef_res = pd.read_csv("outputs/cv_results_Rapamycin-Gefitinib_response.csv")

# create argparser arguments
parser = argparse.ArgumentParser()
# parser.add_argument("model", type=str, default="lgbm")
parser.add_argument("drug", type=str, default=0)
args = parser.parse_args()

PRETRAINED = True
results = []
# for d in DRUGS.values():
res = pd.read_csv(f"outputs/cv_results_{DRUGS[args.drug]}.csv")
# results.append(res)


traindf = pd.read_csv("allTrain.tsv", sep="\t", low_memory=True)
testdf = pd.read_csv("testData.tsv", sep="\t", low_memory=True, index_col=0)
_, _, _, _, X_train, y_train = get_train_test_split(traindf)
X_test_ = prep_test(testdf)
X_test = rename_cols(X_train, X_test_)


def make_model(fsm_str, model_str, mp_str):
    fsm = get_model(fsm_str)
    model = get_model(model_str)
    try:
        mp = ast.literal_eval(mp_str)
    except:
        mp = {}

    return fsm, model, mp


def get_model_info(cv_results, X_train, y_train, label):
    shap_importance = {}
    cs_feats = {}
    model_info = {}
    for fsm_str, model_str, mp_str in cv_results[
        ["feat_sel", "models", "model_params"]
    ].values:
        fsm, model, mp = make_model(fsm_str, model_str, mp_str)
        p = Pipeline(
            [("feats", SelectFromModel(fsm(), max_features=200)), ("clf", model(**mp))]
        )
        p.fit(X_train, y_train[label])
        sv, xt = calc_shap(p, X_train, X_test=None)
        svdf = pd.DataFrame(sv, columns=xt.columns.tolist())
        overall_feats = np.abs(svdf.values).mean(0)
        svdf["c_types"] = traindf["type"].tolist()
        cs_feats_ = svdf.groupby("c_types").mean().idxmax(axis=1)
        for k, v in cs_feats_.to_dict().items():
            if k in cs_feats:
                cs_feats[k].append(v)
            else:
                cs_feats[k] = [v]

        temp_dict = dict(zip(xt.columns.tolist(), overall_feats))
        shap_importance = merge_dictionaries(shap_importance, temp_dict)

    total_features = len(shap_importance)
    cs_feats = find_max_freq_element(cs_feats)
    model_info.update(cs_feats)
    model_info["numer_of_features"] = total_features
    model_info["training_accuracy"] = cv_results["accuracy_x"].mean()
    model_info["Filename"] = f"pipeline_{label}.pkl"
    model_info[
        "algorithm_used"
    ] = f"ensemble of the highest performing models present in cv_results_{label} with accuracy threshold 0.75"
    model_info["feature_eval_method"] = "shapely values"
    model_info_df = pd.DataFrame(
        list(model_info.items()), columns=["Requirements", f"{label}"]
    )

    return model_info_df


def make_predictions(cv_results, X_train, y_train, X_test, label, pretrained=False):
    # weights_ = cv_results["accuracy_x"] / cv_results["accuracy_x"].iloc[0]
    estimators_ = []
    i = 0
    for fsm_str, model_str, mp_str in cv_results[
        ["feat_sel", "models", "model_params"]
    ].values:
        fsm, model, mp = make_model(fsm_str, model_str, mp_str)
        estimators_.append((f"clf_{i}", model(**mp)))
        i += 1
    if not pretrained:
        p = Pipeline(
            [
                ("feats", SelectFromModel(fsm(), max_features=200)),
                (
                    "clf",
                    VotingClassifier(estimators=estimators_, voting="soft"),
                ),
            ]
        )
        p.fit(X_train, y_train[label])
        with open(f"outputs/pipeline_{label}.pkl", "wb") as file:
            pickle.dump(p, file)
    else:
        with open(f"outputs/pipeline_{label}.pkl", "rb") as file:
            p = pickle.load(file)

    predictions = array_to_dataframe(p.predict(X_test), label)
    sv, xt = calc_shap(p, X_train, X_test)
    svdf = pd.DataFrame(sv, columns=xt.columns.tolist())
    top_feats = svdf.idxmax(axis=1)
    predictions[f'{label.split("_")[0]}_key_feature'] = top_feats
    return predictions


model_info = pd.DataFrame()
model_output = pd.DataFrame()
for cvr, label in zip(results, DRUGS.values()):
    fr = filter_results(cvr, thresh=0.85)
    model_info_ = get_model_info(fr, X_train, y_train, label)
    outputs = make_predictions(
        fr, X_train, y_train, X_test, label, pretrained=PRETRAINED
    )
    model_output = pd.concat([model_output, outputs], axis=1)
    model_info = pd.concat([model_info_, model_info], axis=1)
    break
model_output.to_csv("outputs/model_output_.csv")
model_info.to_csv("outputs/model_info_.csv")
