import pandas as pd
import numpy as np
import subprocess
import os

# os.chdir("/notebooks/Scripts")

# the following is the list of all the arguments that we want to run -> (model, embedding, environment, splits, iterations, log, usalgo)lr

args_list = [
    ("xgb", "no", "local", 3, 20, "yes", "tomek"),
    ("lgbm", "no", "local", 3, 20, "yes", "tomek"),
    ("xgb", "no", "local", 3, 20, "yes", "nm"),
    ("lgbm", "no", "local", 3, 20, "yes", "nm"),
    # ("xgb", "no", "local", 3, 20, "yes", "cnn"),
    # ("lgbm", "no", "local", 3, 20, "yes", "cnn"),
    ("xgb", "no", "local", 3, 20, "yes", "rus"),
    ("lgbm", "no", "local", 3, 20, "yes", "rus"),
    ("xgb", "no", "local", 3, 20, "yes", "enn"),
    ("lgbm", "no", "local", 3, 20, "yes", "enn"),
    ("xgb", "no", "local", 3, 20, "yes", "renn"),
    ("lgbm", "no", "local", 3, 20, "yes", "renn"),
    ("xgb", "no", "local", 3, 20, "yes", "allknn"),
    ("lgbm", "no", "local", 3, 20, "yes", "allknn"),
    ("xgb", "no", "local", 3, 20, "yes", "iht"),
    ("lgbm", "no", "local", 3, 20, "yes", "iht"),
    ("xgb", "no", "local", 3, 20, "yes", "ncr"),
    ("lgbm", "no", "local", 3, 20, "yes", "ncr"),
    ("xgb", "no", "local", 3, 20, "yes", "none"),
    ("lgbm", "no", "local", 3, 20, "yes", "none"),
]

# args_list = [
#     ("xgb", "no", "local", 3, 50, "yes", "none"),
#     ("lgbm", "no", "local", 3, 50, "yes", "none"),
#     # ("knn", "no", "local", 3, 50, "yes", "none"),
#     # ("et", "no", "local", 3, 50, "yes", "none"),
#     # ("svc", "no", "local", 3, 20, "yes", "none"),
# ]

for args in args_list:
    subprocess.run(
        "python run_sweep.py "
        + str(args[0])
        + " "
        + str(args[1])
        + " "
        + str(args[2])
        + " "
        + str(args[3])
        + " "
        + str(args[4])
        + " "
        + str(args[5])
        + " "
        + str(args[6]),
        shell=True,
    )
