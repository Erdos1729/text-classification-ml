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

if "WebofScience" not in os.listdir("./data_files"):
    print("\nDownloading web of science data..........")
    download_file("https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/9rw3vkcfy4-2.zip", dest_folder="./data_files")
    print("\nExtracting web of science data from 9rw3vkcfy4-2.zip file ..........")
    all_ext("./data_files/9rw3vkcfy4-2.zip", target_dir="./data_files/WebofScience")
    all_ext("./data_files/WebofScience/WebOfScience.zip", target_dir="./data_files/WebofScience")
else:
    print("\nWeb of science data already downloaded..........")

link = ["./data_files/WebofScience/WOS5736/", "./data_files/WebofScience/WOS11967/", "./data_files/WebofScience/WOS46985/"]

id = []
text = []
label = []
label1 = []
label2 = []
type = []

for lnk in link:
    files = os.listdir(lnk)
    typ = (lnk.replace("./data_files/WebofScience/", "")).replace("/","")
    for n in tqdm(range(len(files))):
        filename = files[n]
        file1 = pd.read_csv(lnk + "/" + str(filename), encoding='latin-1', sep='\t')
        txt = file1.values
        for i in txt:
            if "X.txt" in filename:
                text.append(i[0])
                type.append(typ)
            elif "Y.txt" in filename:
                label.append(i[0])
            elif "YL1.txt" in filename:
                label1.append(i[0])
            elif "YL2.txt" in filename:
                label2.append(i[0])

for i in range(len(text)):
    id.append(i)

df = pd.DataFrame()
df['ID'] = id
df['Text'] = text
df['Label'] = label
df['Label1'] = label1
df['Label2'] = label2
df['Type'] = type

df.to_csv("./data_files/webofscience_data.csv", index=False)
print("\nDownload complete..........")