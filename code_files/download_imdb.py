import warnings
warnings.filterwarnings("ignore")

from keras.datasets import imdb
import pandas as pd
import os
import numpy as np
from tqdm import tqdm

import os
import requests
import shutil

def download_file(url: str, dest_folder: str):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)  # create folder if it does not exist

    filename = url.split('/')[-1].replace(" ", "_")  # be careful with file names
    file_path = os.path.join(dest_folder, filename)

    r = requests.get(url, stream=True)
    if r.ok:
        print("saving to", os.path.abspath(file_path))
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 8):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
    else:  # HTTP status code 4XX/5XX
        print("Download failed: status code {}\n{}".format(r.status_code, r.text))

def all_ext(zip_file: str, target_dir: str):
    shutil.unpack_archive(zip_file, target_dir)

if "aclImdb_v1.tar.gz" not in os.listdir("./data_files"):
    print("\nDownloading IMDB data..........")
    download_file("http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", dest_folder="./data_files")
    print("\nExtracting IMDB data from aclImdb_v1.tar.gz file ..........")
    all_ext("./data_files/aclImdb_v1.tar.gz", target_dir="./data_files")
else:
    print("\nIMDB data already downloaded..........")

link = ["./data_files/aclImdb/train/pos/", "./data_files/aclImdb/train/neg/", "./data_files/aclImdb/test/pos/", "./data_files/aclImdb/test/neg/"]

id = []
rating = []
text = []
label = []
type = []

for lnk in link:
    files = os.listdir(lnk)

    for i in tqdm(range(len(files))):
        filename = files[i]
        file1 = open(lnk + "/" + str(filename), "r", encoding='unicode-escape')
        txt = file1.read()
        filename = filename.replace(".txt", "")
        filename = filename.split("_")
        id.append(filename[0])
        rating.append(filename[1])
        text.append(txt)
        if "pos" in lnk:
            label.append(1)
        else:
            label.append(0)
        if "train" in lnk:
            type.append("train")
        else:
            type.append("test")

df = pd.DataFrame()
df['ID'] = id
df['Rating'] = rating
df['Text'] = text
df['Label'] = label
df['Type'] = type

df.to_csv("./data_files/imdb_data.csv", index=False)
print("\nDownload complete..........")