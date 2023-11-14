import shutil
import os
import gc
import psutil

results_path = r'../results/'
if not os.path.exists(results_path):
    os.makedirs(results_path)

# shutil.copy("README.md", "{}README.md".format(results_path))
shutil.copy("/data/challenge2_files/short_answers.txt", "{}xyz_short_answers.txt.tsv".format(results_path))
# shutil.copytree("sample_submission","{}sample_submission".format(results_path))