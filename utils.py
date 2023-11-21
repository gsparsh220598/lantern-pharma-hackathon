import importlib
import pandas as pd
import shap
from collections import Counter

DRUGS = {
    0: "Rapamycin_response",
    1: "Mitomycin_response",
    2: "Fulvestrant_response",
    3: "Gefitinib_response",
    4: "Rapamycin-Gefitinib_response",
    5: "Mitomycin-Fulvestrant_response",
}


def dict_to_dataframe(input_dict):
    df = pd.DataFrame(
        list(input_dict.items()), columns=["features", "feature_importance"]
    )
    return df


def array_to_dataframe(input_array, column_name="values"):
    df = pd.DataFrame({column_name: input_array})
    return df


def get_model(model_str):
    model_str = model_str.split("'")[1]
    model_ls = model_str.rsplit(".", 1)
    mod = ".".join(model_ls[:-1])
    module = importlib.import_module(mod)
    model_class = getattr(module, model_ls[-1])
    return model_class


def group_results(cv_results):
    cv_results_all = cv_results.groupby(["seed", "feat_sel", "models"])[
        ["accuracy", "f1"]
    ].mean()
    cv_results_all[["accuracy_std", "f1_std"]] = cv_results.groupby(
        ["seed", "feat_sel", "models"]
    )[["accuracy", "f1"]].std()
    cv_results_all = cv_results_all.fillna(0)
    cv_results_all = cv_results_all.sort_values(
        ["accuracy", "accuracy_std"], ascending=False
    )

    cv_results_ = cv_results.groupby(["feat_sel", "models"])[
        ["accuracy", "f1"]
    ].mean()  # .sort_values(['accuracy', 'f1'],ascending=False)
    cv_results_[["accuracy_std", "f1_std"]] = cv_results.groupby(
        ["feat_sel", "models"]
    )[
        ["accuracy", "f1"]
    ].std()  # .sort_values(['accuracy', 'f1'],ascending=False)
    cv_results_ = cv_results_.fillna(0)
    cv_results_ = cv_results_.sort_values(["accuracy", "accuracy_std"], ascending=False)
    return cv_results_all, cv_results_


def filter_results(cv_results, thresh=0.7):
    cv_results_all, _ = group_results(cv_results)
    filtered_res = (
        cv_results_all[cv_results_all["accuracy"] >= thresh]
        .sort_values("accuracy_std")
        .reset_index()
    )
    true_res = cv_results.sort_values("accuracy", ascending=False).reset_index()
    merged_df = pd.merge(
        true_res, filtered_res, on=["seed", "feat_sel", "models"], how="inner"
    )[["feat_sel", "models", "model_params", "accuracy_x"]]
    return merged_df


def calc_shap(pipe, X_train, X_test):
    # pipe.fit(X_train, y_train)
    model = pipe.named_steps["clf"]
    feat_bool = pipe.named_steps["feats"].get_support()
    selected_feats = X_train.iloc[:, feat_bool]
    print(f"------{ pipe['clf'].__class__.__name__ }-----")
    explainer = shap.KernelExplainer(model.predict, selected_feats)
    if X_test is not None:
        shap_values = explainer.shap_values(X_test.iloc[:, feat_bool])
        return shap_values, X_test.iloc[:, feat_bool]
    else:
        shap_values = explainer.shap_values(X_train.iloc[:, feat_bool])
        return shap_values, X_train.iloc[:, feat_bool]


def merge_dictionaries(dict1, dict2):
    counter1 = Counter(dict1)
    counter2 = Counter(dict2)

    merged_counter = counter1 + counter2
    merged_dict = dict(merged_counter)

    return merged_dict


def prep_test(df):
    # df = rename_cols(df, X_train)
    # constant_columns = [col for col in df.columns if df[col].nunique() == 1]
    df.type = pd.Categorical(df.type.astype("category"))
    df.type = df.type.cat.codes
    # df = df.drop(columns=constant_columns, axis=1)
    return df


def rename_cols(X_train, X_test):
    diff_cols = X_train.columns != X_test.columns
    test_cols = X_test.iloc[:, diff_cols].columns.to_list()
    train_cols = X_train.iloc[:, diff_cols].columns.tolist()
    rename_cols = dict(zip(test_cols, train_cols))
    X_test = X_test.rename(rename_cols, axis=1)
    return X_test


def find_max_freq_element(data):
    result = {}
    for key, values in data.items():
        # Use Counter to count the frequency of each element in the list
        counter = Counter(values)

        # Find the element with the highest frequency
        most_common_element, frequency = counter.most_common(1)[0]

        # Store the result in the dictionary
        result[key] = most_common_element
    return result
