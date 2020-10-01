import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from keras.datasets import reuters
import pandas as pd
import os

################################ Select and download data #######################################

print(os.listdir('./data_files'))

datasets = ['reuters_data.csv', 'imdb_data.csv', 'webofscience_data.csv', 'newsgroup']
data = datasets[0]

path = "download_" + data.replace("_data.csv",".py")
print("\n" + "Working on " + data.replace(".csv",""))
time.sleep(5)

if data not in os.listdir('./data_files'):
    if "imdb" in data:
        import download_imdb
        runpy.run_path(path_name=path)
    elif "reuters" in data:
        import download_reuters
        runpy.run_path(path_name=path)
    elif "webofscience_data" in data:
        import download_webscience
        runpy.run_path(path_name=path)
    else:
        print("Download not required for the dataset")
else:
    print("\n" + data.replace(".csv","") + " data downloaded")

################################ Access and clean data for training #######################################

def text_clean(list):
    # write pre-text processing code
    return list

if "newsgroup" in data:
    print("Data directly accessed online")
else:
    df = pd.read_csv('./data_files/' + data)
    text = df['Text'].values.astype(str)
    label = df['Label'].values
    textclean = text_clean(text)
    print("\n", textclean)

################################ Split data for training #######################################

if 'reuters' in data:
    X_train, X_test, y_train, y_test = train_test_split(textclean, label, test_size=0.50)

elif 'imdb' in data:
    is_train = df['Type'] == 'train'
    df_train = df[is_train]
    X_train, y_train = df_train['Text'].values.astype(str), df_train['Label'].values
    is_test = df['Type'] == 'test'
    df_test = df[is_test]
    X_test, y_test = df_test['Text'].values.astype(str), df_test['Label'].values

elif 'webofscience' in data:
    X_train, X_test, y_train, y_test = train_test_split(textclean, label, test_size=0.50)

elif 'news' in data:
    newsgroups_train = fetch_20newsgroups(subset='train')
    newsgroups_test = fetch_20newsgroups(subset='test')
    X_train = newsgroups_train.data
    X_test = newsgroups_test.data
    y_train = newsgroups_train.target
    y_test = newsgroups_test.target
    df = pd.DataFrame()
    y_new = [int(i) for i in y_train] + [int(n) for n in y_test]
    df['Label'] = y_new

################################ Linear SVC model training and output analysis #######################################

svc_mod = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', RandomForestClassifier(n_estimators=100)),
                     ])

svc_mod.fit(X_train, y_train)
predicted = svc_mod.predict(X_test)

# Analyze model metrics
print(metrics.classification_report(y_test, predicted))
print(metrics.confusion_matrix(y_test, predicted))