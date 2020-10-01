
###############################
Text Classification Algorithms
###############################

  
.. figure:: docs/pic/WordArt.png 
 
 
 Referenced paper : `Text Classification Algorithms: A Survey <https://arxiv.org/abs/1904.08067>`__


##################
Table of Contents
##################
.. contents::
  :local:
  :depth: 4

============
Introduction
============

.. figure:: docs/pic/OverviewTextClassification.png 

    
    
====================================
Text and Document Feature Extraction
====================================

----


Text feature extraction and pre-processing for classification algorithms are very significant. In this section, we start to talk about text cleaning since most of documents contain a lot of noise. In this part, we discuss two primary methods of text feature extractions- word embedding and weighted word.


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Text Cleaning and Pre-processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In Natural Language Processing (NLP), most of the text and documents contain many words that are redundant for text classification, such as stopwords, miss-spellings, slangs, and etc. In this section, we briefly explain some techniques and methods for text cleaning and pre-processing text documents. In many algorithms like statistical and probabilistic learning methods, noise and unnecessary features can negatively affect the overall perfomance. So, elimination of these features are extremely important.


-------------
Tokenization
-------------

Tokenization is the process of breaking down a stream of text into words, phrases, symbols, or any other meaningful elements called tokens. The main goal of this step is to extract individual words in a sentence. Along with text classifcation, in text mining, it is necessay to incorporate a parser in the pipeline which performs the tokenization of the documents; for example:

sentence:

.. code::

  After sleeping for four hours, he decided to sleep for another four


In this case, the tokens are as follows:

.. code::

    {'After', 'sleeping', 'for', 'four', 'hours', 'he', 'decided', 'to', 'sleep', 'for', 'another', 'four'}


Here is python code for Tokenization:

.. code:: python

  from nltk.tokenize import word_tokenize
  text = "After sleeping for four hours, he decided to sleep for another four"
  tokens = word_tokenize(text)
  print(tokens)

-----------
Stop words
-----------


Text and document classification over social media, such as Twitter, Facebook, and so on is usually affected by the noisy nature (abbreviations, irregular forms) of the text corpuses.

Here is an exmple from  `geeksforgeeks <https://www.geeksforgeeks.org/removing-stop-words-nltk-python/>`__

.. code:: python

  from nltk.corpus import stopwords
  from nltk.tokenize import word_tokenize

  example_sent = "This is a sample sentence, showing off the stop words filtration."

  stop_words = set(stopwords.words('english'))

  word_tokens = word_tokenize(example_sent)

  filtered_sentence = [w for w in word_tokens if not w in stop_words]

  filtered_sentence = []

  for w in word_tokens:
      if w not in stop_words:
          filtered_sentence.append(w)

  print(word_tokens)
  print(filtered_sentence)



Output:

.. code:: python 

  ['This', 'is', 'a', 'sample', 'sentence', ',', 'showing', 
  'off', 'the', 'stop', 'words', 'filtration', '.']
  ['This', 'sample', 'sentence', ',', 'showing', 'stop',
  'words', 'filtration', '.']


---------------
Capitalization
---------------

Sentences can contain a mixture of uppercase and lower case letters. Multiple sentences make up a text document. To reduce the problem space, the most common approach is to reduce everything to lower case. This brings all words in a document in same space, but it often changes the meaning of some words, such as "US" to "us" where first one represents the United States of America and second one is a pronoun. To solve this, slang and abbreviation converters can be applied.

.. code:: python

  text = "The United States of America (USA) or America, is a federal republic composed of 50 states"
  print(text)
  print(text.lower())

Output:

.. code:: python

  "The United States of America (USA) or America, is a federal republic composed of 50 states"
  "the united states of america (usa) or america, is a federal republic composed of 50 states"

-----------------------
Slangs and Abbreviations
-----------------------

Slangs and abbreviations can cause problems while executing the pre-processing steps. An abbreviation  is a shortened form of a word, such as SVM stand for Support Vector Machine. Slang is a version of language that depicts informal conversation or text that has different meaning, such as "lost the plot", it essentially means that 'they've gone mad'. Common method to deal with these words is converting them to formal language.

---------------
Noise Removal
---------------


Another issue of text cleaning as a pre-processing step is noise removal. Text documents generally contains characters like punctuations or  special characters and they are not necessary for text mining or classification purposes. Although punctuation is critical to understand the meaning of the sentence, but it can affect the classification algorithms negatively.


Here is simple code to remove standard noise from text:


.. code:: python

  def text_cleaner(text):
      rules = [
          {r'>\s+': u'>'},  # remove spaces after a tag opens or closes
          {r'\s+': u' '},  # replace consecutive spaces
          {r'\s*<br\s*/?>\s*': u'\n'},  # newline after a <br>
          {r'</(div)\s*>\s*': u'\n'},  # newline after </p> and </div> and <h1/>...
          {r'</(p|h\d)\s*>\s*': u'\n\n'},  # newline after </p> and </div> and <h1/>...
          {r'<head>.*<\s*(/head|body)[^>]*>': u''},  # remove <head> to </head>
          {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r'\1'},  # show links instead of texts
          {r'[ \t]*<[^<]*?/?>': u''},  # remove remaining tags
          {r'^\s+': u''}  # remove spaces at the beginning
      ]
      for rule in rules:
      for (k, v) in rule.items():
          regex = re.compile(k)
          text = regex.sub(v, text)
      text = text.rstrip()
      return text.lower()
    


-------------------
Spelling Correction
-------------------


An optional part of the pre-processing step is correcting the misspelled words. Different techniques, such as hashing-based and context-sensitive spelling correction techniques, or  spelling correction using trie and damerau-levenshtein distance bigram have been introduced to tackle this issue.


.. code:: python

  from autocorrect import spell

  print spell('caaaar')
  print spell(u'mussage')
  print spell(u'survice')
  print spell(u'hte')

Result:

.. code::

    caesar
    message
    service
    the


------------
Stemming
------------


Text Stemming is modifying a word to obtain its variants using different linguistic processeses like affixation (addition of affixes). For example, the stem of the word "studying" is "study", to which -ing.


Here is an example of Stemming from `NLTK <https://pythonprogramming.net/stemming-nltk-tutorial/>`__

.. code:: python

    from nltk.stem import PorterStemmer
    from nltk.tokenize import sent_tokenize, word_tokenize

    ps = PorterStemmer()

    example_words = ["python","pythoner","pythoning","pythoned","pythonly"]
    
    for w in example_words:
    print(ps.stem(w))


Result:

.. code::

  python
  python
  python
  python
  pythonli

-------------
Lemmatization
-------------


Text lemmatization is the process of eliminating redundant prefix or suffix of a word and extract the base word (lemma).


.. code:: python

  from nltk.stem import WordNetLemmatizer

  lemmatizer = WordNetLemmatizer()

  print(lemmatizer.lemmatize("cats"))

~~~~~~~~~~~~~~
Word Embedding
~~~~~~~~~~~~~~

Different word embedding procedures have been proposed to translate these unigrams into consummable input for machine learning algorithms. A very simple way to perform such embedding is term-frequency~(TF) where each word will be mapped to a number corresponding to the number of occurrence of that word in the whole corpora. The other term frequency functions have been also used that represent word-frequency as Boolean or logarithmically scaled number. Here, each document will be converted to a vector of same length containing the frequency of the words in that document. Although such approach may seem very intuitive but it suffers from the fact that particular words that are used very commonly in language literature might dominate this sort of word representations.

.. image:: docs/pic/CBOW.png


--------
Word2Vec
--------

Original from https://code.google.com/p/word2vec/

I’ve copied it to a github project so that I can apply and track community
patches (starting with capability for Mac OS X
compilation).

-  **makefile and some source has been modified for Mac OS X
   compilation** See
   https://code.google.com/p/word2vec/issues/detail?id=1#c5
-  **memory patch for word2vec has been applied** See
   https://code.google.com/p/word2vec/issues/detail?id=2
-  Project file layout altered

There seems to be a segfault in the compute-accuracy utility.

To get started:

::

   cd scripts && ./demo-word.sh

Original README text follows:

This tool provides an efficient implementation of the continuous bag-of-words and skip-gram architectures for computing vector representations of words. These representations can be subsequently used in many natural language processing applications and for further research purposes. 


this code provides an implementation of the Continuous Bag-of-Words (CBOW) and
the Skip-gram model (SG), as well as several demo scripts.

Given a text corpus, the word2vec tool learns a vector for every word in
the vocabulary using the Continuous Bag-of-Words or the Skip-Gram neural
network architectures. The user should specify the following: -
desired vector dimensionality (size of the context window for
either the Skip-Gram or the Continuous Bag-of-Words model),  training
algorithm (hierarchical softmax and / or negative sampling), threshold
for downsampling the frequent words, number of threads to use,
format of the output word vector file (text or binary).

Usually, other hyper-parameters, such as the learning rate do not
need to be tuned for different training sets.

The script demo-word.sh downloads a small (100MB) text corpus from the
web, and trains a small word vector model. After the training is
finished, users can interactively explore the similarity of the
words.

More information about the scripts is provided at
https://code.google.com/p/word2vec/


~~~~~~~~~~~~~~
Weighted Words
~~~~~~~~~~~~~~


--------------
Term frequency
--------------

Term frequency is Bag of words that is one of the simplest techniques of text feature extraction. This method is based on counting number of the words in each document and assign it to feature space.


-----------------------------------------
Term Frequency-Inverse Document Frequency
-----------------------------------------
The mathematical representation of weight of a term in a document by Tf-idf is given:

.. image:: docs/eq/tf-idf.gif
   :width: 10px
   
Where N is number of documents and df(t) is the number of documents containing the term t in the corpus. The first part would improve recall and the later would improve the precision of the word embedding. Although tf-idf tries to overcome the problem of common terms in document, it still suffers from some other descriptive limitations. Namely, tf-idf cannot account for the similarity between words in the document since each word is presented as an index. In the recent years, with development of more complex models, such as neural nets, new methods has been presented that can incorporate concepts, such as similarity of words and part of speech tagging. This work uses, word2vec and Glove, two of the most common methods that have been successfully used for deep learning techniques.


.. code:: python

    from sklearn.feature_extraction.text import TfidfVectorizer
    def loadData(X_train, X_test,MAX_NB_WORDS=75000):
        vectorizer_x = TfidfVectorizer(max_features=MAX_NB_WORDS)
        X_train = vectorizer_x.fit_transform(X_train).toarray()
        X_test = vectorizer_x.transform(X_test).toarray()
        print("tf-idf with",str(np.array(X_train).shape[1]),"features")
        return (X_train,X_test)
   
   
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Comparison of Feature Extraction Techniques
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


+---------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------+
|                **Model**              |                                                                        **Advantages**                                                                    |                                                   **Limitation**                                               |
+---------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------+
|            **Weighted Words**         |  * Easy to compute                                                                                                                                       |  * It does not capture the position in the text (syntactic)                                                    |
|                                       |                                                                                                                                                          |                                                                                                                |
|                                       |  * Easy to compute the similarity between 2 documents using it                                                                                           |  * It does not capture meaning in the text (semantics)                                                         |
|                                       |                                                                                                                                                          |                                                                                                                |
|                                       |  * Basic metric to extract the most descriptive terms in a document                                                                                      |                                                                                                                |
|                                       |                                                                                                                                                          |  * Common words effect on the results (e.g., “am”, “is”, etc.)                                                 |
|                                       |  * Works with an unknown word (e.g., New words in languages)                                                                                             |                                                                                                                |
+---------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------+
|            **TF-IDF**                 |  * Easy to compute                                                                                                                                       |  * It does not capture the position in the text (syntactic)                                                    |
|                                       |                                                                                                                                                          |                                                                                                                |
|                                       |                                                                                                                                                          |                                                                                                                |
|                                       |  * Easy to compute the similarity between 2 documents using it                                                                                           |  * It does not capture meaning in the text (semantics)                                                         |
|                                       |                                                                                                                                                          |                                                                                                                |
|                                       |                                                                                                                                                          |                                                                                                                |
|                                       |  * Basic metric to extract the most descriptive terms in a document                                                                                      |                                                                                                                |
|                                       |                                                                                                                                                          |                                                                                                                |
|                                       |                                                                                                                                                          |                                                                                                                |
|                                       |  * Common words do not affect the results due to IDF (e.g., “am”, “is”, etc.)                                                                            |                                                                                                                |
+---------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------+
|               **Word2Vec**            |  * It captures the position of the words in the text (syntactic)                                                                                         |  * It cannot capture the meaning of the word from the text (fails to capture polysemy)                         |
|                                       |                                                                                                                                                          |                                                                                                                |
|                                       |  * It captures meaning in the words (semantics)                                                                                                          |  * It cannot capture out-of-vocabulary words from corpus                                                       |                                                                                             |  * Computationally is more expensive in comparing with GloVe and Word2Vec                                      |
+---------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------+
|**Contextualized Word Representations**|  * It captures the meaning of the word from the text (incorporates context, handling polysemy)                                                           |  * Memory consumption for storage                                                                              |
|                                       |                                                                                                                                                          |                                                                                                                |
|                                       |                                                                                                                                                          |  * Improves performance notably on downstream tasks. Computationally is more expensive in comparison to others |
|                                       |                                                                                                                                                          |                                                                                                                |
|                                       |                                                                                                                                                          |  * Needs another word embedding for all LSTM and feedforward layers                                            |
|                                       |                                                                                                                                                          |                                                                                                                |
|                                       |                                                                                                                                                          |  * It cannot capture out-of-vocabulary words from a corpus                                                     |
|                                       |                                                                                                                                                          |                                                                                                                |
|                                       |                                                                                                                                                          |                                                                                                                |
|                                       |                                                                                                                                                          |  * Works only sentence and document level (it cannot work for individual word level)                           |
+---------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------+


========================
Dimensionality Reduction
========================

----

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Principal Component Analysis (PCA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Principle component analysis~(PCA) is the most popular technique in multivariate analysis and dimensionality reduction. PCA is a method to identify a subspace in which the data approximately lies. This means finding new variables that are uncorrelated and maximizing the variance to preserve as much variability as possible.


Example of PCA on text dataset (20newsgroups) from  tf-idf with 75000 features to 2000 components:

.. code:: python

    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np

    def TFIDF(X_train, X_test, MAX_NB_WORDS=75000):
        vectorizer_x = TfidfVectorizer(max_features=MAX_NB_WORDS)
        X_train = vectorizer_x.fit_transform(X_train).toarray()
        X_test = vectorizer_x.transform(X_test).toarray()
        print("tf-idf with", str(np.array(X_train).shape[1]), "features")
        return (X_train, X_test)


    from sklearn.datasets import fetch_20newsgroups

    newsgroups_train = fetch_20newsgroups(subset='train')
    newsgroups_test = fetch_20newsgroups(subset='test')
    X_train = newsgroups_train.data
    X_test = newsgroups_test.data
    y_train = newsgroups_train.target
    y_test = newsgroups_test.target

    X_train,X_test = TFIDF(X_train,X_test)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2000)
    X_train_new = pca.fit_transform(X_train)
    X_test_new = pca.transform(X_test)

    print("train with old features: ",np.array(X_train).shape)
    print("train with new features:" ,np.array(X_train_new).shape)
    
    print("test with old features: ",np.array(X_test).shape)
    print("test with new features:" ,np.array(X_test_new).shape)

output:

.. code:: python

    tf-idf with 75000 features
    train with old features:  (11314, 75000)
    train with new features: (11314, 2000)
    test with old features:  (7532, 75000)
    test with new features: (7532, 2000)



~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Linear Discriminant Analysis (LDA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Linear Discriminant Analysis (LDA) is another commonly used technique for data classification and dimensionality reduction. LDA is particularly helpful where the within-class frequencies are unequal and their performances have been evaluated on randomly generated test data. Class-dependent and class-independent transformation are two approaches in LDA where the ratio of between-class-variance to within-class-variance and the ratio of the overall-variance to within-class-variance are used respectively. 



.. code:: python


  from sklearn.feature_extraction.text import TfidfVectorizer
  import numpy as np
  from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


  def TFIDF(X_train, X_test, MAX_NB_WORDS=75000):
      vectorizer_x = TfidfVectorizer(max_features=MAX_NB_WORDS)
      X_train = vectorizer_x.fit_transform(X_train).toarray()
      X_test = vectorizer_x.transform(X_test).toarray()
      print("tf-idf with", str(np.array(X_train).shape[1]), "features")
      return (X_train, X_test)


  from sklearn.datasets import fetch_20newsgroups

  newsgroups_train = fetch_20newsgroups(subset='train')
  newsgroups_test = fetch_20newsgroups(subset='test')
  X_train = newsgroups_train.data
  X_test = newsgroups_test.data
  y_train = newsgroups_train.target
  y_test = newsgroups_test.target

  X_train,X_test = TFIDF(X_train,X_test)



  LDA = LinearDiscriminantAnalysis(n_components=15)
  X_train_new = LDA.fit(X_train,y_train)
  X_train_new =  LDA.transform(X_train)
  X_test_new = LDA.transform(X_test)

  print("train with old features: ",np.array(X_train).shape)
  print("train with new features:" ,np.array(X_train_new).shape)

  print("test with old features: ",np.array(X_test).shape)
  print("test with new features:" ,np.array(X_test_new).shape)


output:

.. code:: 

    tf-idf with 75000 features
    train with old features:  (11314, 75000)
    train with new features: (11314, 15)
    test with old features:  (7532, 75000)
    test with new features: (7532, 15)
    
    
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Non-negative Matrix Factorization (NMF)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. code:: python


    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    from sklearn.decomposition import NMF


    def TFIDF(X_train, X_test, MAX_NB_WORDS=75000):
        vectorizer_x = TfidfVectorizer(max_features=MAX_NB_WORDS)
        X_train = vectorizer_x.fit_transform(X_train).toarray()
        X_test = vectorizer_x.transform(X_test).toarray()
        print("tf-idf with", str(np.array(X_train).shape[1]), "features")
        return (X_train, X_test)


    from sklearn.datasets import fetch_20newsgroups

    newsgroups_train = fetch_20newsgroups(subset='train')
    newsgroups_test = fetch_20newsgroups(subset='test')
    X_train = newsgroups_train.data
    X_test = newsgroups_test.data
    y_train = newsgroups_train.target
    y_test = newsgroups_test.target

    X_train,X_test = TFIDF(X_train,X_test)



    NMF_ = NMF(n_components=2000)
    X_train_new = NMF_.fit(X_train)
    X_train_new =  NMF_.transform(X_train)
    X_test_new = NMF_.transform(X_test)

    print("train with old features: ",np.array(X_train).shape)
    print("train with new features:" ,np.array(X_train_new).shape)

    print("test with old features: ",np.array(X_test).shape)
    print("test with new features:" ,np.array(X_test_new))

output:

.. code:: 

    tf-idf with 75000 features
    train with old features:  (11314, 75000)
    train with new features: (11314, 2000)
    test with old features:  (7532, 75000)
    test with new features: (7532, 2000)
    
    

~~~~~~~~~~~~~~~~~
Random Projection
~~~~~~~~~~~~~~~~~
Random projection or random feature is a dimensionality reduction technique mostly used for very large volume dataset or very high dimensional feature space. Text and document, especially with weighted feature extraction, can contain a huge number of underlying features.
Many researchers addressed Random Projection for text data for text mining, text classification and/or dimensionality reduction.
We start to review some random projection techniques. 


.. image:: docs/pic/Random%20Projection.png

.. code:: python

    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np

    def TFIDF(X_train, X_test, MAX_NB_WORDS=75000):
        vectorizer_x = TfidfVectorizer(max_features=MAX_NB_WORDS)
        X_train = vectorizer_x.fit_transform(X_train).toarray()
        X_test = vectorizer_x.transform(X_test).toarray()
        print("tf-idf with", str(np.array(X_train).shape[1]), "features")
        return (X_train, X_test)


    from sklearn.datasets import fetch_20newsgroups

    newsgroups_train = fetch_20newsgroups(subset='train')
    newsgroups_test = fetch_20newsgroups(subset='test')
    X_train = newsgroups_train.data
    X_test = newsgroups_test.data
    y_train = newsgroups_train.target
    y_test = newsgroups_test.target

    X_train,X_test = TFIDF(X_train,X_test)

    from sklearn import random_projection

    RandomProjection = random_projection.GaussianRandomProjection(n_components=2000)
    X_train_new = RandomProjection.fit_transform(X_train)
    X_test_new = RandomProjection.transform(X_test)

    print("train with old features: ",np.array(X_train).shape)
    print("train with new features:" ,np.array(X_train_new).shape)

    print("test with old features: ",np.array(X_test).shape)
    print("test with new features:" ,np.array(X_test_new).shape)

output:

.. code:: python

    tf-idf with 75000 features
    train with old features:  (11314, 75000)
    train with new features: (11314, 2000)
    test with old features:  (7532, 75000)
    test with new features: (7532, 2000)
    
~~~~~~~~~~~
Autoencoder
~~~~~~~~~~~


Autoencoder is a neural network technique that is trained to attempt to map its input to its output. The autoencoder as dimensional reduction methods have achieved great success via the powerful reprehensibility of neural networks. The main idea is, one hidden layer between the input and output layers with fewer neurons can be used to reduce the dimension of feature space. Specially for texts, documents, and sequences that contains many features, autoencoder could help to process data faster and more efficiently.


.. image:: docs/pic/Autoencoder.png



.. code:: python

  from keras.layers import Input, Dense
  from keras.models import Model

  # this is the size of our encoded representations
  encoding_dim = 1500  

  # this is our input placeholder
  input = Input(shape=(n,))
  # "encoded" is the encoded representation of the input
  encoded = Dense(encoding_dim, activation='relu')(input)
  # "decoded" is the lossy reconstruction of the input
  decoded = Dense(n, activation='sigmoid')(encoded)

  # this model maps an input to its reconstruction
  autoencoder = Model(input, decoded)

  # this model maps an input to its encoded representation
  encoder = Model(input, encoded)
  

  encoded_input = Input(shape=(encoding_dim,))
  # retrieve the last layer of the autoencoder model
  decoder_layer = autoencoder.layers[-1]
  # create the decoder model
  decoder = Model(encoded_input, decoder_layer(encoded_input))
  
  autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
  
  

Load data:


.. code:: python

  autoencoder.fit(x_train, x_train,
                  epochs=50,
                  batch_size=256,
                  shuffle=True,
                  validation_data=(x_test, x_test))
                  

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
T-distributed Stochastic Neighbor Embedding (T-SNE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



T-distributed Stochastic Neighbor Embedding (T-SNE) is a nonlinear dimensionality reduction technique for embedding high-dimensional data which is mostly used for visualization in a low-dimensional space. This approach is based on `G. Hinton and ST. Roweis <https://www.cs.toronto.edu/~fritz/absps/sne.pdf>`__ . SNE works by converting the high dimensional Euclidean distances into conditional probabilities which represent similarities.

 `Example <http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html>`__:


.. code:: python

   import numpy as np
   from sklearn.manifold import TSNE
   X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
   X_embedded = TSNE(n_components=2).fit_transform(X)
   X_embedded.shape


Example of Glove and T-SNE for text:

.. image:: docs/pic/TSNE.png

===============================
Text Classification Techniques
===============================

----


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Rocchio classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first version of Rocchio algorithm is introduced by rocchio in 1971 to use relevance feedback in querying full-text databases. Since then many researchers have addressed and developed this technique for text and document classification. This method uses TF-IDF weights for each informative word instead of a set of Boolean features. Using a training set of documents, Rocchio's algorithm builds a prototype vector for each class which is an average vector over all training document vectors that belongs to a certain class. Then, it will assign each test document to a class with maximum similarity that between test document and each of the prototype vectors.


When in nearest centroid classifier, we used for text as input data for classification with tf-idf vectors, this classifier is known as the Rocchio classifier.

.. code:: python

    from sklearn.neighbors.nearest_centroid import NearestCentroid
    from sklearn.pipeline import Pipeline
    from sklearn import metrics
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.datasets import fetch_20newsgroups

    newsgroups_train = fetch_20newsgroups(subset='train')
    newsgroups_test = fetch_20newsgroups(subset='test')
    X_train = newsgroups_train.data
    X_test = newsgroups_test.data
    y_train = newsgroups_train.target
    y_test = newsgroups_test.target

    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', NearestCentroid()),
                         ])

    text_clf.fit(X_train, y_train)


    predicted = text_clf.predict(X_test)

    print(metrics.classification_report(y_test, predicted))




Output:

.. code:: python

                  precision    recall  f1-score   support

              0       0.75      0.49      0.60       319
              1       0.44      0.76      0.56       389
              2       0.75      0.68      0.71       394
              3       0.71      0.59      0.65       392
              4       0.81      0.71      0.76       385
              5       0.83      0.66      0.74       395
              6       0.49      0.88      0.63       390
              7       0.86      0.76      0.80       396
              8       0.91      0.86      0.89       398
              9       0.85      0.79      0.82       397
             10       0.95      0.80      0.87       399
             11       0.94      0.66      0.78       396
             12       0.40      0.70      0.51       393
             13       0.84      0.49      0.62       396
             14       0.89      0.72      0.80       394
             15       0.55      0.73      0.63       398
             16       0.68      0.76      0.71       364
             17       0.97      0.70      0.81       376
             18       0.54      0.53      0.53       310
             19       0.58      0.39      0.47       251

    avg / total       0.74      0.69      0.70      7532



~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Boosting and Bagging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

---------
Boosting
---------

.. image:: docs/pic/Boosting.PNG


**Boosting** is a Ensemble learning meta-algorithm for primarily reducing variance in supervised learning. It is basically a family of machine learning algorithms that convert weak learners to strong ones. Boosting is based on the question posed by `Michael Kearns <https://en.wikipedia.org/wiki/Michael_Kearns_(computer_scientist)>`__  and Leslie Valiant (1988, 1989) Can a set of weak learners create a single strong learner? A weak learner is defined to be a Classification that is only slightly correlated with the true classification (it can label examples better than random guessing). In contrast, a strong learner is a classifier that is arbitrarily well-correlated with the true classification.




.. code:: python

  from sklearn.ensemble import GradientBoostingClassifier
  from sklearn.pipeline import Pipeline
  from sklearn import metrics
  from sklearn.feature_extraction.text import CountVectorizer
  from sklearn.feature_extraction.text import TfidfTransformer
  from sklearn.datasets import fetch_20newsgroups

  newsgroups_train = fetch_20newsgroups(subset='train')
  newsgroups_test = fetch_20newsgroups(subset='test')
  X_train = newsgroups_train.data
  X_test = newsgroups_test.data
  y_train = newsgroups_train.target
  y_test = newsgroups_test.target

  text_clf = Pipeline([('vect', CountVectorizer()),
                       ('tfidf', TfidfTransformer()),
                       ('clf', GradientBoostingClassifier(n_estimators=100)),
                       ])

  text_clf.fit(X_train, y_train)


  predicted = text_clf.predict(X_test)

  print(metrics.classification_report(y_test, predicted))


Output:
 
.. code:: python

               precision    recall  f1-score   support
            0       0.81      0.66      0.73       319
            1       0.69      0.70      0.69       389
            2       0.70      0.68      0.69       394
            3       0.64      0.72      0.68       392
            4       0.79      0.79      0.79       385
            5       0.83      0.64      0.72       395
            6       0.81      0.84      0.82       390
            7       0.84      0.75      0.79       396
            8       0.90      0.86      0.88       398
            9       0.90      0.85      0.88       397
           10       0.93      0.86      0.90       399
           11       0.90      0.81      0.85       396
           12       0.33      0.69      0.45       393
           13       0.87      0.72      0.79       396
           14       0.87      0.84      0.85       394
           15       0.85      0.87      0.86       398
           16       0.65      0.78      0.71       364
           17       0.96      0.74      0.84       376
           18       0.70      0.55      0.62       310
           19       0.62      0.56      0.59       251

  avg / total       0.78      0.75      0.76      7532

  
-------
Bagging
-------

.. image:: docs/pic/Bagging.PNG


.. code:: python

    from sklearn.ensemble import BaggingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn import metrics
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.datasets import fetch_20newsgroups

    newsgroups_train = fetch_20newsgroups(subset='train')
    newsgroups_test = fetch_20newsgroups(subset='test')
    X_train = newsgroups_train.data
    X_test = newsgroups_test.data
    y_train = newsgroups_train.target
    y_test = newsgroups_test.target

    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', BaggingClassifier(KNeighborsClassifier())),
                         ])

    text_clf.fit(X_train, y_train)


    predicted = text_clf.predict(X_test)

    print(metrics.classification_report(y_test, predicted))


Output:
 
.. code:: python

               precision    recall  f1-score   support
            0       0.57      0.74      0.65       319
            1       0.60      0.56      0.58       389
            2       0.62      0.54      0.58       394
            3       0.54      0.57      0.55       392
            4       0.63      0.54      0.58       385
            5       0.68      0.62      0.65       395
            6       0.55      0.46      0.50       390
            7       0.77      0.67      0.72       396
            8       0.79      0.82      0.80       398
            9       0.74      0.77      0.76       397
           10       0.81      0.86      0.83       399
           11       0.74      0.85      0.79       396
           12       0.67      0.49      0.57       393
           13       0.78      0.51      0.62       396
           14       0.76      0.78      0.77       394
           15       0.71      0.81      0.76       398
           16       0.73      0.73      0.73       364
           17       0.64      0.79      0.71       376
           18       0.45      0.69      0.54       310
           19       0.61      0.54      0.57       251

  avg / total       0.67      0.67      0.67      7532
  


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Naive Bayes Classifier
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Naïve Bayes text classification has been used in industry
and academia for a long time (introduced by Thomas Bayes
between 1701-1761). However, this technique
is being studied since the 1950s for text and document categorization. Naive Bayes Classifier (NBC) is generative
model which is widely used in Information Retrieval. Many researchers addressed and developed this technique
for their applications. We start with the most basic version
of NBC which developed by using term-frequency (Bag of
Word) fetaure extraction technique by counting number of
words in documents


.. code:: python

    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    from sklearn import metrics
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.datasets import fetch_20newsgroups

    newsgroups_train = fetch_20newsgroups(subset='train')
    newsgroups_test = fetch_20newsgroups(subset='test')
    X_train = newsgroups_train.data
    X_test = newsgroups_test.data
    y_train = newsgroups_train.target
    y_test = newsgroups_test.target

    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB()),
                         ])

    text_clf.fit(X_train, y_train)


    predicted = text_clf.predict(X_test)

    print(metrics.classification_report(y_test, predicted))
 
 
Output:
 
.. code:: python

                   precision    recall  f1-score   support

              0       0.80      0.52      0.63       319
              1       0.81      0.65      0.72       389
              2       0.82      0.65      0.73       394
              3       0.67      0.78      0.72       392
              4       0.86      0.77      0.81       385
              5       0.89      0.75      0.82       395
              6       0.93      0.69      0.80       390
              7       0.85      0.92      0.88       396
              8       0.94      0.93      0.93       398
              9       0.92      0.90      0.91       397
             10       0.89      0.97      0.93       399
             11       0.59      0.97      0.74       396
             12       0.84      0.60      0.70       393
             13       0.92      0.74      0.82       396
             14       0.84      0.89      0.87       394
             15       0.44      0.98      0.61       398
             16       0.64      0.94      0.76       364
             17       0.93      0.91      0.92       376
             18       0.96      0.42      0.58       310
             19       0.97      0.14      0.24       251

    avg / total       0.82      0.77      0.77      7532


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
K-nearest Neighbor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
R
In machine learning, the k-nearest neighbors algorithm (kNN)
is a non-parametric technique used for classification.
This method is used in Natural-language processing (NLP)
as a text classification technique in many researches in the past
decades.

.. image:: docs/pic/KNN.png

.. code:: python

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn import metrics
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.datasets import fetch_20newsgroups

    newsgroups_train = fetch_20newsgroups(subset='train')
    newsgroups_test = fetch_20newsgroups(subset='test')
    X_train = newsgroups_train.data
    X_test = newsgroups_test.data
    y_train = newsgroups_train.target
    y_test = newsgroups_test.target

    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', KNeighborsClassifier()),
                         ])

    text_clf.fit(X_train, y_train)

    predicted = text_clf.predict(X_test)

    print(metrics.classification_report(y_test, predicted))

Output:

.. code:: python

                   precision    recall  f1-score   support

              0       0.43      0.76      0.55       319
              1       0.50      0.61      0.55       389
              2       0.56      0.57      0.57       394
              3       0.53      0.58      0.56       392
              4       0.59      0.56      0.57       385
              5       0.69      0.60      0.64       395
              6       0.58      0.45      0.51       390
              7       0.75      0.69      0.72       396
              8       0.84      0.81      0.82       398
              9       0.77      0.72      0.74       397
             10       0.85      0.84      0.84       399
             11       0.76      0.84      0.80       396
             12       0.70      0.50      0.58       393
             13       0.82      0.49      0.62       396
             14       0.79      0.76      0.78       394
             15       0.75      0.76      0.76       398
             16       0.70      0.73      0.72       364
             17       0.62      0.76      0.69       376
             18       0.55      0.61      0.58       310
             19       0.56      0.49      0.52       251

    avg / total       0.67      0.66      0.66      7532






~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Support Vector Machine (SVM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


The original version of SVM was introduced by Vapnik and  Chervonenkis in 1963. The early 1990s, nonlinear version was addressed by BE. Boser et al.. Original version of SVM was designed for binary classification problem, but Many researchers have worked on multi-class problem using this authoritative technique.


The advantages of support vector machines are based on scikit-learn page:

* Effective in high dimensional spaces.
* Still effective in cases where number of dimensions is greater than the number of samples.
* Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.
* Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.


The disadvantages of support vector machines include:

* If the number of features is much greater than the number of samples, avoiding over-fitting via choosing kernel functions and regularization term is crucial.
* SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).



.. image:: docs/pic/SVM.png


.. code:: python


    from sklearn.svm import LinearSVC
    from sklearn.pipeline import Pipeline
    from sklearn import metrics
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.datasets import fetch_20newsgroups

    newsgroups_train = fetch_20newsgroups(subset='train')
    newsgroups_test = fetch_20newsgroups(subset='test')
    X_train = newsgroups_train.data
    X_test = newsgroups_test.data
    y_train = newsgroups_train.target
    y_test = newsgroups_test.target

    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', LinearSVC()),
                         ])

    text_clf.fit(X_train, y_train)


    predicted = text_clf.predict(X_test)

    print(metrics.classification_report(y_test, predicted))


output:


.. code:: python

                   precision    recall  f1-score   support

              0       0.82      0.80      0.81       319
              1       0.76      0.80      0.78       389
              2       0.77      0.73      0.75       394
              3       0.71      0.76      0.74       392
              4       0.84      0.86      0.85       385
              5       0.87      0.76      0.81       395
              6       0.83      0.91      0.87       390
              7       0.92      0.91      0.91       396
              8       0.95      0.95      0.95       398
              9       0.92      0.95      0.93       397
             10       0.96      0.98      0.97       399
             11       0.93      0.94      0.93       396
             12       0.81      0.79      0.80       393
             13       0.90      0.87      0.88       396
             14       0.90      0.93      0.92       394
             15       0.84      0.93      0.88       398
             16       0.75      0.92      0.82       364
             17       0.97      0.89      0.93       376
             18       0.82      0.62      0.71       310
             19       0.75      0.61      0.68       251

    avg / total       0.85      0.85      0.85      7532






~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Decision Tree
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One of earlier classification algorithm for text and data mining is decision tree. Decision tree classifiers (DTC's) are used successfully in many diverse areas of classification. The structure of this technique includes a hierarchical decomposition of the data space (only train dataset). Decision tree as classification task was introduced by `D. Morgan <http://www.aclweb.org/anthology/P95-1037>`__ and developed by `JR. Quinlan <https://courses.cs.ut.ee/2009/bayesian-networks/extras/quinlan1986.pdf>`__. The main idea is creating trees based on the attributes of the data points, but the challenge is determining which attribute should be in parent level and which one should be in child level. To solve this problem, `De Mantaras <https://link.springer.com/article/10.1023/A:1022694001379>`__ introduced statistical modeling for feature selection in tree.


.. code:: python

    from sklearn import tree
    from sklearn.pipeline import Pipeline
    from sklearn import metrics
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.datasets import fetch_20newsgroups

    newsgroups_train = fetch_20newsgroups(subset='train')
    newsgroups_test = fetch_20newsgroups(subset='test')
    X_train = newsgroups_train.data
    X_test = newsgroups_test.data
    y_train = newsgroups_train.target
    y_test = newsgroups_test.target

    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', tree.DecisionTreeClassifier()),
                         ])

    text_clf.fit(X_train, y_train)


    predicted = text_clf.predict(X_test)

    print(metrics.classification_report(y_test, predicted))


output:


.. code:: python

                   precision    recall  f1-score   support

              0       0.51      0.48      0.49       319
              1       0.42      0.42      0.42       389
              2       0.51      0.56      0.53       394
              3       0.46      0.42      0.44       392
              4       0.50      0.56      0.53       385
              5       0.50      0.47      0.48       395
              6       0.66      0.73      0.69       390
              7       0.60      0.59      0.59       396
              8       0.66      0.72      0.69       398
              9       0.53      0.55      0.54       397
             10       0.68      0.66      0.67       399
             11       0.73      0.69      0.71       396
             12       0.34      0.33      0.33       393
             13       0.52      0.42      0.46       396
             14       0.65      0.62      0.63       394
             15       0.68      0.72      0.70       398
             16       0.49      0.62      0.55       364
             17       0.78      0.60      0.68       376
             18       0.38      0.38      0.38       310
             19       0.32      0.32      0.32       251

    avg / total       0.55      0.55      0.55      7532



~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Random Forest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Random forests or random decision forests technique is an ensemble learning method for text classification. This method was introduced by `T. Kam Ho <https://doi.org/10.1109/ICDAR.1995.598994>`__ in 1995 for first time which used t trees in parallel. This technique was later developed by `L. Breiman <https://link.springer.com/article/10.1023/A:1010933404324>`__ in 1999 that they found converged for RF as a margin measure.


.. image:: docs/pic/RF.png

.. code:: python

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn import metrics
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.datasets import fetch_20newsgroups

    newsgroups_train = fetch_20newsgroups(subset='train')
    newsgroups_test = fetch_20newsgroups(subset='test')
    X_train = newsgroups_train.data
    X_test = newsgroups_test.data
    y_train = newsgroups_train.target
    y_test = newsgroups_test.target

    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', RandomForestClassifier(n_estimators=100)),
                         ])

    text_clf.fit(X_train, y_train)


    predicted = text_clf.predict(X_test)

    print(metrics.classification_report(y_test, predicted))


output:


.. code:: python


                    precision    recall  f1-score   support

              0       0.69      0.63      0.66       319
              1       0.56      0.69      0.62       389
              2       0.67      0.78      0.72       394
              3       0.67      0.67      0.67       392
              4       0.71      0.78      0.74       385
              5       0.78      0.68      0.73       395
              6       0.74      0.92      0.82       390
              7       0.81      0.79      0.80       396
              8       0.90      0.89      0.90       398
              9       0.80      0.89      0.84       397
             10       0.90      0.93      0.91       399
             11       0.89      0.91      0.90       396
             12       0.68      0.49      0.57       393
             13       0.83      0.65      0.73       396
             14       0.81      0.88      0.84       394
             15       0.68      0.91      0.78       398
             16       0.67      0.86      0.75       364
             17       0.93      0.78      0.85       376
             18       0.86      0.48      0.61       310
             19       0.79      0.31      0.45       251

    avg / total       0.77      0.76      0.75      7532




~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Conditional Random Field (CRF)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Conditional Random Field (CRF) is an undirected graphical model as shown in figure. CRFs state the conditional probability of a label sequence *Y* give a sequence of observation *X* *i.e.* P(Y|X). CRFs can incorporate complex features of observation sequence without violating the independence assumption by modeling the conditional probability of the label sequences rather than the joint probability P(X,Y). The concept of clique which is a fully connected subgraph and clique potential are used for computing P(X|Y). Considering one potential function for each clique of the graph, the probability of a variable configuration corresponds to the product of a series of non-negative potential function. The value computed by each potential function is equivalent to the probability of the variables in its corresponding clique taken on a particular configuration.


.. image:: docs/pic/CRF.png


Example from `Here <http://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html>`__
Let’s use CoNLL 2002 data to build a NER system
CoNLL2002 corpus is available in NLTK. We use Spanish data.


.. code:: python

      import nltk
      import sklearn_crfsuite
      from sklearn_crfsuite import metrics
      nltk.corpus.conll2002.fileids()
      train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))
      test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))
      
      
sklearn-crfsuite (and python-crfsuite) supports several feature formats; here we use feature dicts.

.. code:: python

      def word2features(sent, i):
          word = sent[i][0]
          postag = sent[i][1]

          features = {
              'bias': 1.0,
              'word.lower()': word.lower(),
              'word[-3:]': word[-3:],
              'word[-2:]': word[-2:],
              'word.isupper()': word.isupper(),
              'word.istitle()': word.istitle(),
              'word.isdigit()': word.isdigit(),
              'postag': postag,
              'postag[:2]': postag[:2],
          }
          if i > 0:
              word1 = sent[i-1][0]
              postag1 = sent[i-1][1]
              features.update({
                  '-1:word.lower()': word1.lower(),
                  '-1:word.istitle()': word1.istitle(),
                  '-1:word.isupper()': word1.isupper(),
                  '-1:postag': postag1,
                  '-1:postag[:2]': postag1[:2],
              })
          else:
              features['BOS'] = True

          if i < len(sent)-1:
              word1 = sent[i+1][0]
              postag1 = sent[i+1][1]
              features.update({
                  '+1:word.lower()': word1.lower(),
                  '+1:word.istitle()': word1.istitle(),
                  '+1:word.isupper()': word1.isupper(),
                  '+1:postag': postag1,
                  '+1:postag[:2]': postag1[:2],
              })
          else:
              features['EOS'] = True

          return features


      def sent2features(sent):
          return [word2features(sent, i) for i in range(len(sent))]

      def sent2labels(sent):
          return [label for token, postag, label in sent]

      def sent2tokens(sent):
          return [token for token, postag, label in sent]

      X_train = [sent2features(s) for s in train_sents]
      y_train = [sent2labels(s) for s in train_sents]

      X_test = [sent2features(s) for s in test_sents]
      y_test = [sent2labels(s) for s in test_sents]


To see all possible CRF parameters check its docstring. Here we are useing L-BFGS training algorithm (it is default) with Elastic Net (L1 + L2) regularization.



.. code:: python

      crf = sklearn_crfsuite.CRF(
          algorithm='lbfgs',
          c1=0.1,
          c2=0.1,
          max_iterations=100,
          all_possible_transitions=True
      )
      crf.fit(X_train, y_train)


Evaluation


.. code:: python

      y_pred = crf.predict(X_test)
      print(metrics.flat_classification_report(
          y_test, y_pred,  digits=3
      ))


Output:

.. code:: python

                     precision    recall  f1-score   support

            B-LOC      0.810     0.784     0.797      1084
           B-MISC      0.731     0.569     0.640       339
            B-ORG      0.807     0.832     0.820      1400
            B-PER      0.850     0.884     0.867       735
            I-LOC      0.690     0.637     0.662       325
           I-MISC      0.699     0.589     0.639       557
            I-ORG      0.852     0.786     0.818      1104
            I-PER      0.893     0.943     0.917       634
                O      0.992     0.997     0.994     45355

      avg / total      0.970     0.971     0.971     51533



~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Comparison Text Classification Algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| **Model**                          | **Advantages**                                                                                                                                           | **Disadvantages**                                                                                                                       |
+------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| **Rocchio Algorithm**              |  * Easy to implement                                                                                                                                     |  * The user can only retrieve a few relevant documents                                                                                  |
|                                    |                                                                                                                                                          |                                                                                                                                         |
|                                    |  * Computationally is very cheap                                                                                                                         |  * Rocchio often misclassifies the type for multimodal class                                                                            |
|                                    |                                                                                                                                                          |                                                                                                                                         |
|                                    |  * Relevance feedback mechanism (benefits to ranking documents as  not relevant)                                                                         |  * This techniques is not very robust                                                                                                   |
|                                    |                                                                                                                                                          |                                                                                                                                         |
|                                    |                                                                                                                                                          |  * linear combination in this algorithm is not good for multi-class datasets                                                            |
+------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| **Boosting and Bagging**           |  * Improves the stability and accuracy (takes the advantage of ensemble learning where in multiple weak learner outperform a single strong learner.)     |  * Computational complexity                                                                                                             |
|                                    |                                                                                                                                                          |                                                                                                                                         |
|                                    |  * Reducing variance which helps to avoid overfitting problems.                                                                                          |  * loss of interpretability (if the number of models is hight, understanding the model is very difficult)                               |
|                                    |                                                                                                                                                          |                                                                                                                                         |
|                                    |                                                                                                                                                          |  * Requires careful tuning of different hyper-parameters.                                                                               |
+------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| **Logistic Regression**            |  * Easy to implement                                                                                                                                     |  * it cannot solve non-linear problems                                                                                                  |
|                                    |                                                                                                                                                          |                                                                                                                                         |
|                                    |  * does not require too many computational resources                                                                                                     |  * prediction requires that each data point be independent                                                                              |
|                                    |                                                                                                                                                          |                                                                                                                                         |
|                                    |  * it does not require input features to be scaled (pre-processing)                                                                                      |  * attempting to predict outcomes based on a set of independent variables                                                               |
|                                    |                                                                                                                                                          |                                                                                                                                         |
|                                    |  * It does not require any tuning                                                                                                                        |                                                                                                                                         |
+------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| **Naive Bayes Classifier**         |  * It works very well with text data                                                                                                                     |  *  A strong assumption about the shape of the data distribution                                                                        |
|                                    |                                                                                                                                                          |                                                                                                                                         |
|                                    |  * Easy to implement                                                                                                                                     |  * limited by data scarcity for which any possible value in feature space, a likelihood value must be estimated by a frequentist        |
|                                    |                                                                                                                                                          |                                                                                                                                         |
|                                    |  * Fast in comparing to other algorithms                                                                                                                 |                                                                                                                                         |
+------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| **K-Nearest Neighbor**             |  * Effective for text datasets                                                                                                                           |  * computational of this model is very expensive                                                                                        |
|                                    |                                                                                                                                                          |                                                                                                                                         |
|                                    |  * non-parametric                                                                                                                                        |  * diffcult to find optimal value of k                                                                                                  |
|                                    |                                                                                                                                                          |                                                                                                                                         |
|                                    |  * More local characteristics of text or document are considered                                                                                         |  * Constraint for large search problem to find nearest neighbors                                                                        |
|                                    |                                                                                                                                                          |                                                                                                                                         |
|                                    |  * Naturally handles multi-class datasets                                                                                                                |  * Finding a meaningful distance function is difficult for text datasets                                                                |
+------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| **Support Vector Machine (SVM)**   |  * SVM can model non-linear decision boundaries                                                                                                          |  * lack of transparency in results caused by a high number of dimensions (especially for text data).                                    |
|                                    |                                                                                                                                                          |                                                                                                                                         |
|                                    |  * Performs similarly to logistic regression when linear separation                                                                                      |  * Choosing an efficient kernel function is difficult (Susceptible to overfitting/training issues depending on kernel)                  |
|                                    |                                                                                                                                                          |                                                                                                                                         |
|                                    |  * Robust against overfitting problems~(especially for text dataset due to high-dimensional space)                                                       |  * Memory complexity                                                                                                                    |
+------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| **Decision Tree**                  |  * Can easily handle qualitative (categorical) features                                                                                                  |  * Issues with diagonal decision boundaries                                                                                             |
|                                    |                                                                                                                                                          |                                                                                                                                         |
|                                    |  * Works well with decision boundaries parellel to the feature axis                                                                                      |  * Can be easily overfit                                                                                                                |
|                                    |                                                                                                                                                          |                                                                                                                                         |
|                                    |  * Decision tree is a very fast algorithm for both learning and prediction                                                                               |  * extremely sensitive to small perturbations in the data                                                                               |
|                                    |                                                                                                                                                          |                                                                                                                                         |
|                                    |                                                                                                                                                          |  * Problems with out-of-sample prediction                                                                                               |
+------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| **Conditional Random Field (CRF)** |  * Its feature design is flexible                                                                                                                        |  * High computational complexity of the training step                                                                                   |
|                                    |                                                                                                                                                          |                                                                                                                                         |
|                                    |  * Since CRF computes the conditional probability of global optimal output nodes, it overcomes the drawbacks of label bias                               |  * this algorithm does not perform with unknown words                                                                                   |
|                                    |                                                                                                                                                          |                                                                                                                                         |
|                                    |  * Combining the advantages of classification and graphical modeling which combining the ability to compactly model multivariate data                    |  * Problem about online learning (It makes it very difficult to re-train the model when newer data becomes available.)                  |
+------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| **Random Forest**                  |  * Ensembles of decision trees are very fast to train in comparison to other techniques                                                                  |  * Quite slow to create predictions once trained                                                                                        |
|                                    |                                                                                                                                                          |                                                                                                                                         |
|                                    |  * Reduced variance (relative to regular trees)                                                                                                          |  * more trees in forest increases time complexity in the prediction step                                                                |
|                                    |                                                                                                                                                          |                                                                                                                                         |
|                                    |  * Not require preparation and pre-processing of the input data                                                                                          |  * Not as easy to visually interpret                                                                                                    |
|                                    |                                                                                                                                                          |                                                                                                                                         |
|                                    |                                                                                                                                                          |  * Overfitting can easily occur                                                                                                         |
|                                    |                                                                                                                                                          |                                                                                                                                         |
|                                    |                                                                                                                                                          |  * Need to choose the number of trees at forest                                                                                         |
+------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------+


==========
Evaluation
==========

----

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
F1 Score
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. image:: docs/pic/F1.png

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Matthew correlation coefficient (MCC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Compute the Matthews correlation coefficient (MCC)

The Matthews correlation coefficient is used in machine learning as a measure of the quality of binary (two-class) classification problems. It takes into account of true and false positives and negatives and is generally regarded as a balanced measure which can be used even if the classes are of very different sizes. The MCC is in essence a correlation coefficient value between -1 and +1. A coefficient of +1 represents a perfect prediction, 0 an average random prediction and -1 an inverse prediction. The statistic is also known as the phi coefficient. 


.. code:: python

    from sklearn.metrics import matthews_corrcoef
    y_true = [+1, +1, +1, -1]
    y_pred = [+1, -1, +1, +1]
    matthews_corrcoef(y_true, y_pred)  



~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Receiver operating characteristics (ROC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ROC curves are typically used in binary classification to study the output of a classifier. In order to extend ROC curve and ROC area to multi-class or multi-label classification, it is necessary to binarize the output. One ROC curve can be drawn per label, but one can also draw a ROC curve by considering each element of the label indicator matrix as a binary prediction (micro-averaging).

Another evaluation measure for multi-class classification is macro-averaging, which gives equal weight to the classification of each label. [`sources  <http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html>`__] 

.. code:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from itertools import cycle

    from sklearn import svm, datasets
    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import label_binarize
    from sklearn.multiclass import OneVsRestClassifier
    from scipy import interp

    # Import some data to play with
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Binarize the output
    y = label_binarize(y, classes=[0, 1, 2])
    n_classes = y.shape[1]

    # Add noisy features to make the problem harder
    random_state = np.random.RandomState(0)
    n_samples, n_features = X.shape
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                        random_state=0)

    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                     random_state=random_state))
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
   


Plot of a ROC curve for a specific class


.. code:: python

    plt.figure()
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


.. image:: /docs/pic/sphx_glr_plot_roc_001.png


~~~~~~~~~~~~~~~~~~~~~~~
Area Under Curve (AUC)
~~~~~~~~~~~~~~~~~~~~~~~

Area  under ROC curve (AUC) is a summary metric that measures the entire area underneath the ROC curve. AUC holds helpful properties, such as  increased  sensitivity in the analysis of variance (ANOVA) tests, independence of decision threshold, invariance to a priori class probability and the indication of how well negative and positive classes are regarding decision index.


.. code:: python

      import numpy as np
      from sklearn import metrics
      fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
      metrics.auc(fpr, tpr)




==========================
Text and Document Datasets
==========================

----

~~~~~
IMDB
~~~~~

- `IMDB Dataset <http://ai.stanford.edu/~amaas/data/sentiment/>`__
- `Download Link <http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz>`__

This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. We provide a set of 25,000 highly polar movie reviews for training, and 25,000 for testing. There is additional unlabeled data for use as well. Raw text and already processed bag of words formats are provided. See the README file contained in the release for more details. Dataset of 25,000 movies reviews from IMDB, labeled by sentiment (positive/negative). Reviews have been preprocessed, and each review is encoded as a sequence of word indexes (integers). For convenience, words are indexed by overall frequency in the dataset, so that for instance the integer "3" encodes the 3rd most frequent word in the data. This allows for quick filtering operations, such as "only consider the top 10,000 most common words, but eliminate the top 20 most common words".

As a convention, "0" does not stand for a specific word, but instead is used to encode any unknown word.


.. code:: python


  from keras.datasets import imdb

  (x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",
                                                        num_words=None,
                                                        skip_top=0,
                                                        maxlen=None,
                                                        seed=113,
                                                        start_char=1,
                                                        oov_char=2,
                                                        index_from=3)

get_word_index function               

.. code:: python

  tf.keras.datasets.imdb.get_word_index(path="imdb_word_index.json")


~~~~~~~~~~~~~
Reuters-21578
~~~~~~~~~~~~~

- `Reters-21578 Dataset <https://keras.io/datasets/>`__


Dataset of 11,228 newswires from Reuters, labeled over 46 topics. As with the IMDB dataset, each wire is encoded as a sequence of word indexes (same conventions).


.. code:: python

  from keras.datasets import reuters

  (x_train, y_train), (x_test, y_test) = reuters.load_data(path="reuters.npz",
                                                           num_words=None,
                                                           skip_top=0,
                                                           maxlen=None,
                                                           test_split=0.2,
                                                           seed=113,
                                                           start_char=1,
                                                           oov_char=2,
                                                           index_from=3)

get_word_index function               

.. code:: python

  tf.keras.datasets.reuters.get_word_index(path="reuters_word_index.json")

                                                         
~~~~~~~~~~~~~
20Newsgroups
~~~~~~~~~~~~~

- `20Newsgroups Dataset <https://archive.ics.uci.edu/ml/datasets/Twenty+Newsgroups>`__
- `Download Link <https://archive.ics.uci.edu/ml/machine-learning-databases/20newsgroups-mld/20_newsgroups.tar.gz>`__

The 20 newsgroups dataset comprises around 18000 newsgroups posts on 20 topics split in two subsets: one for training (or development) and the other one for testing (or for performance evaluation). The split between the train and test set is based upon messages posted before and after a specific date.

This module contains two loaders. The first one, sklearn.datasets.fetch_20newsgroups, returns a list of the raw texts that can be fed to text feature extractors, such as sklearn.feature_extraction.text.CountVectorizer with custom parameters so as to extract feature vectors. The second one, sklearn.datasets.fetch_20newsgroups_vectorized, returns ready-to-use features, i.e., it is not necessary to use a feature extractor.


.. code:: python

  from sklearn.datasets import fetch_20newsgroups
  newsgroups_train = fetch_20newsgroups(subset='train')

  from pprint import pprint
  pprint(list(newsgroups_train.target_names))
  
  ['alt.atheism',
   'comp.graphics',
   'comp.os.ms-windows.misc',
   'comp.sys.ibm.pc.hardware',
   'comp.sys.mac.hardware',
   'comp.windows.x',
   'misc.forsale',
   'rec.autos',
   'rec.motorcycles',
   'rec.sport.baseball',
   'rec.sport.hockey',
   'sci.crypt',
   'sci.electronics',
   'sci.med',
   'sci.space',
   'soc.religion.christian',
   'talk.politics.guns',
   'talk.politics.mideast',
   'talk.politics.misc',
   'talk.religion.misc']
 
 
~~~~~~~~~~~~~~~~~~~~~~
Web of Science Dataset
~~~~~~~~~~~~~~~~~~~~~~

Description of Dataset:
- `Web of Science Dataset <https://data.mendeley.com/datasets/9rw3vkcfy4/2>`__
- `Download Link <https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/9rw3vkcfy4-2.zip`>`__

Here is three datasets which include WOS-11967 , WOS-46985, and WOS-5736
Each folder contains:

- X.txt
- Y.txt
- YL1.txt
- YL2.txt

X is input data that include text sequences
Y is target value
YL1 is target value of level one (parent label)
YL2 is target value of level one (child label)

Meta-data:
This folder contain on data file as following attribute:
Y1 Y2 Y Domain area keywords Abstract

Abstract is input data that include text sequences of 46,985 published paper
Y is target value
YL1 is target value of level one (parent label)
YL2 is target value of level one (child label)
Domain is majaor domain which include 7 labales: {Computer Science,Electrical Engineering, Psychology, Mechanical Engineering,Civil Engineering, Medical Science, biochemistry}
area is subdomain or area of the paper, such as CS-> computer graphics which contain 134 labels.
keywords : is authors keyword of the papers

-  Web of Science Dataset `WOS-11967 <http://dx.doi.org/10.17632/9rw3vkcfy4.2>`__
..

  This dataset contains 11,967 documents with 35 categories which include 7 parents categories.

-  Web of Science Dataset `WOS-46985 <http://dx.doi.org/10.17632/9rw3vkcfy4.2>`__
      
..

  This dataset contains 46,985 documents with 134 categories which include 7 parents categories.

-  Web of Science Dataset `WOS-5736 <http://dx.doi.org/10.17632/9rw3vkcfy4.2>`__

..
  
  This dataset contains 5,736 documents with 11 categories which include 3 parents categories.
     
     
================================
Text Classification Applications
================================


----



~~~~~~~~~~~~~~~~~~~~~~
Information Retrieval
~~~~~~~~~~~~~~~~~~~~~~
Information retrieval is finding documents of an unstructured data that meet an information need from within large collections of documents. With the rapid growth of online information, particularly in text format, text classification has become a  significant technique for managing this type of data. Some of the important methods used in this area are Naive Bayes, SVM, decision tree, J48, k-NN and IBK. One of the most challenging applications for document and text dataset processing is applying document categorization methods for information retrieval.

- 🎓 `Introduction to information retrieval <http://eprints.bimcoordinator.co.uk/35/>`__ Manning, C., Raghavan, P., & Schütze, H. (2010).
     
- 🎓 `Web forum retrieval and text analytics: A survey <http://www.nowpublishers.com/article/Details/INR-062>`__ Hoogeveen, Doris, et al.. (2018).

- 🎓 `Automatic Text Classification in Information retrieval: A Survey <https://dl.acm.org/citation.cfm?id=2905191>`__ Dwivedi, Sanjay K., and Chandrakala Arya.. (2016).

~~~~~~~~~~~~~~~~~~~~~~
Information Filtering
~~~~~~~~~~~~~~~~~~~~~~
Information filtering refers to selection of relevant information or rejection of irrelevant information from a stream of incoming data. Information filtering systems are typically used to measure and forecast users' long-term interests. Probabilistic models, such as Bayesian inference network, are commonly used in information filtering systems. Bayesian inference networks employ recursive inference to propagate values through the inference network and return documents with the highest ranking. Chris used vector space model with iterative refinement for filtering task.
 

- 🎓 `Search engines: Information retrieval in practice <http://library.mpib-berlin.mpg.de/toc/z2009_2465.pdf/>`__ Croft, W. B., Metzler, D., & Strohman, T. (2010).

- 🎓 `Implementation of the SMART information retrieval system <https://ecommons.cornell.edu/bitstream/handle/1813/6526/85-686.pdf?sequence=1>`__ Buckley, Chris

~~~~~~~~~~~~~~~~~~~~~~
Sentiment Analysis
~~~~~~~~~~~~~~~~~~~~~~
Sentiment analysis is a computational approach toward identifying opinion, sentiment, and subjectivity in text. Sentiment classification methods classify a document associated with an opinion to be positive or negative. The assumption is that document d is expressing an opinion on a single entity e and opinions are formed via a single opinion holder h. Naive Bayesian classification and SVM are some of the most popular supervised learning methods that have been used for sentiment classification. Features such as terms and their respective frequency, part of speech, opinion words and phrases, negations and syntactic dependency have been used in sentiment classification techniques.

- 🎓 `Opinion mining and sentiment analysis <http://www.nowpublishers.com/article/Details/INR-011>`__ Pang, Bo, and Lillian Lee. (2008).

- 🎓 `A survey of opinion mining and sentiment analysis <https://link.springer.com/chapter/10.1007/978-1-4614-3223-4_13>`__ Liu, Bing, and Lei Zhang. (2010).

- 🎓 `Thumbs up?: sentiment classification using machine learning techniques <https://dl.acm.org/citation.cfm?id=1118704>`__ Pang, Bo, Lillian Lee, and Shivakumar Vaithyanathan. 

~~~~~~~~~~~~~~~~~~~~~~
Recommender Systems
~~~~~~~~~~~~~~~~~~~~~~
Content-based recommender systems suggest items to users based on the description of an item and a profile of the user's interests. 
A user's profile can be learned from user feedback (history of the search queries or self reports) on items as well as self-explained features~(filter or conditions on the queries) in one's profile. 
In this way, input to such recommender systems can be semi-structured such that some attributes are extracted from free-text field while others are directly specified. Many different types of text classification methods, such as decision trees, nearest neighbor methods, Rocchio's algorithm, linear classifiers, probabilistic methods, and Naive Bayes, have been used to model user's preference.

- 🎓 `Content-based recommender systems <https://link.springer.com/chapter/10.1007/978-3-319-29659-3_4>`__ Aggarwal, Charu C. (2016).

- 🎓 `Content-based recommendation systems <https://link.springer.com/chapter/10.1007/978-3-540-72079-9_10>`__ Pazzani, Michael J., and Daniel Billsus.

~~~~~~~~~~~~~~~~~~~~~~
Knowledge Management
~~~~~~~~~~~~~~~~~~~~~~
Textual databases are significant sources of information and knowledge. A large percentage of corporate information (nearly 80 %) exists in textual data formats (unstructured). In knowledge distillation, patterns or knowledge are inferred from immediate forms that can be semi-structured ( e.g.conceptual graph representation) or structured/relational data representation). A given intermediate form can be document-based such that each entity represents an object or concept of interest in a particular domain. Document categorization is one of the most common methods for mining document-based intermediate forms. In the other work, text classification has been used to find the relationship between railroad accidents' causes and their correspondent descriptions in reports.

- 🎓 `Text mining: concepts, applications, tools and issues-an overview <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.403.2426&rep=rep1&type=pdf>`__ Sumathy, K. L., and M. Chidambaram.  (2013).

~~~~~~~~~~~~~~~~~~~~~~
Document Summarization
~~~~~~~~~~~~~~~~~~~~~~
Text classification used for document summarizing which summary of a document may employ words or phrases which do not appear in the original document.  Multi-document summarization also is necessitated due to increasing online information rapidly. So, many researchers focus on this task using text classification to extract important feature out of a document.

- 🎓 `Advances in automatic text summarization <https://books.google.com/books?hl=en&lr=&id=YtUZQaKDmzEC&oi=fnd&pg=PA215&dq=Advances+in+automatic+text+summarization&ots=ZpvCsrG-dC&sig=8ecTDTrQR4mMzDnKvI58sowh3Fg>`__ Mani, Inderjeet. 

- 🎓 `Improving Multi-Document Summarization via Text Classification. <https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/viewPaper/14525>`__ Cao, Ziqiang, et al. (2017).

================================
Text Classification Support
================================

~~~~~~~~~~~~~~~~~~~~~~
Health
~~~~~~~~~~~~~~~~~~~~~~
Most textual information in the medical domain is presented in an unstructured or narrative form with ambiguous terms and typographical errors. Such information needs to be available instantly throughout the patient-physicians encounters in different stages of diagnosis and treatment. Medical coding, which consists of assigning medical diagnoses to specific class values obtained from a large set of categories, is an area of healthcare applications where text classification techniques can be highly valuable. In the other research, J. Zhang et al. introduced Patient2Vec, to learn an interpretable deep representation of longitudinal electronic health record (EHR) data which is personalized for each patient. Patient2Vec is a novel technique of text dataset feature embedding that can learn a personalized interpretable deep representation of EHR data based on recurrent neural networks and the attention mechanism. Text classification has also been applied in the development of Medical Subject Headings (MeSH) and Gene Ontology (GO). 


- 🎓 `Patient2Vec: A Personalized Interpretable Deep Representation of the Longitudinal Electronic Health Record <https://ieeexplore.ieee.org/abstract/document/8490816/>`__ Zhang, Jinghe, et al. (2018)

- 🎓 `Combining Bayesian text classification and shrinkage to automate healthcare coding: A data quality analysis <https://dl.acm.org/citation.cfm?id=2063506>`__ Lauría, Eitel JM, and Alan D. March. (2011).

- 🎓 `A <http://b/>`__ c. (2010).

- 🎓 `MeSH Up: effective MeSH text classification for improved document retrieval <https://academic.oup.com/bioinformatics/article-abstract/25/11/1412/333120>`__ Trieschnigg, Dolf, et al.

~~~~~~~~~~~~~~~~~~~~~~
Social Sciences
~~~~~~~~~~~~~~~~~~~~~~
Text classification and document categorization has increasingly been applied to understanding human behavior in past decades. Recent data-driven efforts in human behavior research have focused on mining language contained in informal notes and text datasets, including short message service (SMS), clinical notes, social media, etc. These studies have mostly focused on using approaches based on frequencies of word occurrence (i.e. how often a word appears in a document) or features based on Linguistic Inquiry Word Count (LIWC), a well-validated lexicon of categories of words with psychological relevance.

- 🎓 `Identification of imminent suicide risk among young adults using text messages <https://dl.acm.org/citation.cfm?id=3173987>`__ Nobles, Alicia L., et al. (2018).

- 🎓 `Textual Emotion Classification: An Interoperability Study on Cross-Genre Data Sets <https://link.springer.com/chapter/10.1007/978-3-319-63004-5_21>`__ Ofoghi, Bahadorreza, and Karin Verspoor. (2017).

- 🎓 `Social Monitoring for Public Health <https://www.morganclaypool.com/doi/abs/10.2200/S00791ED1V01Y201707ICR060>`__ Paul, Michael J., and Mark Dredze (2017).

~~~~~~~~~~~~~~~~~~~~~~
Business and Marketing
~~~~~~~~~~~~~~~~~~~~~~
profitable companies and organizations are progressively using social media for marketing purposes. Opening mining from social media such as Facebook, Twitter, and so on is main target of companies to rapidly increase their profits. Text and documents classification is a powerful tool for companies to find their customers easier than ever.  

- 🎓 `Opinion mining using ensemble text hidden Markov models for text classification <https://www.sciencedirect.com/science/article/pii/S0957417417304979>`__ Kang, Mangi, Jaelim Ahn, and Kichun Lee. (2018).

- 🎓 `Classifying business marketing messages on Facebook <https://www.researchgate.net/profile/Bei_Yu2/publication/236246670_Classifying_Business_Marketing_Messages_on_Facebook/links/56bcb34408ae6cc737c6335b.pdf>`__ Yu, Bei, and Linchi Kwok.

~~~~~~~~~~~~~~~~~~~~~~
Law
~~~~~~~~~~~~~~~~~~~~~~
Huge volumes of legal text information and documents have been generated by governmental institutions. Retrieving this information and automatically classifying it can not only help lawyers but also their clients.
In the United States, the law is derived from five sources: constitutional law, statutory law, treaties, administrative regulations, and the common law. Also, many new legal documents are created each year. Categorization of these documents is the main challenge of the lawyer community.

- 🎓 `Represent yourself in court: How to prepare & try a winning case <https://books.google.com/books?hl=en&lr=&id=-lodDQAAQBAJ&oi=fnd&pg=PP1&dq=Represent+yourself+in+court:+How+to+prepare+%5C%26+try+a+winning+case&ots=tgJ8Q2MkH_&sig=9o3ILDn3LfO30BZKsyI2Ou7Q8Qs>`__ Bergman, Paul, and Sara J. Berman. (2016)

- 🎓 `Text retrieval in the legal world <https://link.springer.com/article/10.1007/BF00877694>`__ Turtle, Howard.

==========
Citations:
==========

----

.. code::

    @ARTICLE{Kowsari2018Text_Classification,
        title={Text Classification Algorithms: A Survey},
        author={Kowsari, Kamran and Jafari Meimandi, Kiana and Heidarysafa, Mojtaba and Mendu, Sanjana and Barnes, Laura E. and Brown, Donald E.},
        journal={Information},
        VOLUME = {10},  
        YEAR = {2019},
        NUMBER = {4},
        ARTICLE-NUMBER = {150},
        URL = {http://www.mdpi.com/2078-2489/10/4/150},
        ISSN = {2078-2489},
        publisher={Multidisciplinary Digital Publishing Institute}
    }

.. |RMDL| image:: http://kowsari.net/onewebmedia/RMDL.jpg
.. |line| image:: docs/pic/line.png
          :alt: Foo
.. |HDLTex| image:: http://kowsari.net/____impro/1/onewebmedia/HDLTex.png?etag=W%2F%22c90cd-59c4019b%22&sourceContentType=image%2Fpng&ignoreAspectRatio&resize=821%2B326&extract=0%2B0%2B821%2B325?raw=false


.. |twitter| image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social
    :target: https://twitter.com/intent/tweet?text=Text%20Classification%20Algorithms:%20A%20Survey%0aGitHub:&url=https://github.com/Erdos1729/text-classification-ml&hashtags=Text_Classification,classification,MachineLearning,Categorization,NLP,NATURAL,LANGUAGE,PROCESSING
    
.. |contributions-welcome| image:: https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat
    :target: https://github.com/Erdos1729/text-classification-ml/pulls
.. |ansicolortags| image:: https://img.shields.io/pypi/l/ansicolortags.svg
      :target: https://github.com/Erdos1729/text-classification-ml/blob/master/LICENSE
.. |contributors| image:: https://img.shields.io/github/contributors/Erdos1729/text-classification-ml.svg
      :target: https://github.com/Erdos1729/text-classification-ml/graphs/contributors 

.. |arXiv| image:: https://img.shields.io/badge/arXiv-1904.08067-red.svg?style=flat
   :target: https://arxiv.org/abs/1904.08067
   
.. |DOI| image:: https://img.shields.io/badge/DOI-10.3390/info10040150-blue.svg?style=flat
   :target: https://doi.org/10.3390/info10040150
   
   
.. |medium| image:: https://img.shields.io/badge/Medium-Text%20Classification-blueviolet.svg
    :target: https://medium.com/text-classification-algorithms/text-classification-algorithms-a-survey-a215b7ab7e2d
    
.. |mendeley| image:: https://img.shields.io/badge/Mendeley-Add%20to%20Library-critical.svg
    :target: https://www.mendeley.com/import/?url=https://doi.org/10.3390/info10040150