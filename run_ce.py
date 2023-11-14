import pandas as pd
import numpy as np
import subprocess

args_list = [
    (10000, 15000),
    (15000, 20000),
    (20000, 25000),
    (25000, 30000),
    (30000, 35000),
    (35000, 40000),
    (40000, 45000),
    (45000, 50000),
    (50000, 55000),
    (55000, 60000),
    (60000, 65000),
    (65000, 70000),
    (70000, 75000),
    (75000, 79747),
]

args_list = [(0, 10)]

for args in args_list:
    subprocess.run(
        "python Scripts/create_embeddings.py " + str(args[0]) + " " + str(args[1]),
        shell=True,
    )
