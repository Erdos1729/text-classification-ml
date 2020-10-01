import warnings
warnings.filterwarnings("ignore")

from keras.datasets import reuters
import pandas as pd
import os
import numpy as np

index = reuters.get_word_index(path="reuters_word_index.json")
print("Please find the word index below")
print("\n", index)

print("\nDownloading Reuters data..........")
(X_train, y_train), (X_test, y_test) = reuters.load_data(path="reuters.npz",
                                                         num_words=None,
                                                         skip_top=0,
                                                         maxlen=None,
                                                         test_split=0.2,
                                                         seed=113,
                                                         start_char=1,
                                                         oov_char=2,
                                                         index_from=3)


list_new = []
def convert_num_to_words(list, list_new):
    from tqdm import tqdm
    for n in tqdm(range(len(list))):
        list_x = []
        for i in range(len(list[n])):
            for key, value in index.items():
                if list[n][i] == value:
                    list_x.append(key)
        list_new.append(list_x)
    return list_new

def listToString(s):
    str1 = ""
    for ele in s:
        str1 += ele
    return str1

def list_transform(list):
    for i in range(len(list)):
        list[i] = " ".join(str(n) for n in list[i])
    list = [listToString(i) for i in list]
    return list

xtrain_new = []
xtest_new = []
convert_num_to_words(X_train, xtrain_new)
convert_num_to_words(X_test, xtest_new)
list_transform(xtrain_new)
list_transform(xtest_new)

print(len(xtrain_new), " ", len(xtest_new))

df = pd.DataFrame()
df['Text'] = xtrain_new + xtest_new
print(df['Text'])
df['Label'] = list(y_train) + list(y_test)
print(len(df['Label']), " ",len(y_train), " ",len(y_test))
df.to_csv('./data_files/reuters_data.csv', index=False)
print("\nDownload complete..........")