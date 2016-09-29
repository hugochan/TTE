import os
from shutil import copyfile
import numpy as np

# copyfile(src, dst)

select_k = 100
src = "linux/corpus/20news-bydate/20news-bydate-train/"
dst = "linux/corpus/20news-bydate/20news-t20-d100/"

dirs = os.listdir(src)[1:]

for each_dir in dirs:
    path = os.path.join(src, each_dir)
    files = [filename for filename in os.listdir(path) if os.path.isfile(os.path.join(path, filename))]
    selected_files = np.random.choice(files, min(select_k, len(files)), replace=False)
    for each_file in selected_files:
        copyfile(os.path.join(path, each_file), os.path.join(dst, each_file))
