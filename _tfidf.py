"""
what if the probability of masking is a function of tfidf score?
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import random
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
import string
from nltk.corpus import stopwords as stop_words


class WhiteSpacePreprocessing():
    
    def __init__(self, documents, stopwords_language="english", vocabulary_size=2000):
        self.documents = documents
        self.stopwords = set(stop_words.words(stopwords_language))
        self.vocabulary_size = vocabulary_size

        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn("WhiteSpacePreprocessing is deprecated and will be removed in future versions."
                      "Use WhiteSpacePreprocessingStopwords.")

    def preprocess(self):
        preprocessed_docs_tmp = self.documents
        preprocessed_docs_tmp = [doc.lower() for doc in preprocessed_docs_tmp]
        preprocessed_docs_tmp = [doc.translate(
            str.maketrans(string.punctuation, ' ' * len(string.punctuation))) for doc in preprocessed_docs_tmp]
        preprocessed_docs_tmp = [' '.join([w for w in doc.split() if len(w) > 0 and w not in self.stopwords])
                                 for doc in preprocessed_docs_tmp]

        vectorizer = CountVectorizer(max_features=self.vocabulary_size)
        vectorizer.fit_transform(preprocessed_docs_tmp)
        temp_vocabulary = set(vectorizer.get_feature_names())
        
        # filter only the words included in the top 2000 vocabulary? (why?)

        preprocessed_docs_tmp = [' '.join([w for w in doc.split() if w in temp_vocabulary])
                                 for doc in preprocessed_docs_tmp]

        preprocessed_docs, unpreprocessed_docs, retained_indices = [], [], []
        for i, doc in enumerate(preprocessed_docs_tmp):
            if len(doc) > 0:
                preprocessed_docs.append(doc)
                unpreprocessed_docs.append(self.documents[i])
                retained_indices.append(i)

        vocabulary = list(set([item for doc in preprocessed_docs for item in doc.split()]))

        return preprocessed_docs, unpreprocessed_docs, vocabulary, retained_indices

class CTFIDFVectorizer(TfidfTransformer):
    def __init__(self, *args, **kwargs):
        super(CTFIDFVectorizer, self).__init__(*args, **kwargs)

    def fit(self, X: sp.csr_matrix, n_samples: int):
        """Learn the idf vector (global term weights) """
        # X: count matrix
        # n_samples: 
        _, n_features = X.shape
        df = np.squeeze(np.asarray(X.sum(axis=0)))
        # sum all count of terms across documents
        
        idf = np.log(n_samples / df)
        self._idf_diag = sp.diags(idf, offsets=0,
                                  shape=(n_features, n_features),
                                  format='csr',
                                  dtype=np.float64)
        return self

    def transform(self, X: sp.csr_matrix) -> sp.csr_matrix:
        """Transform a count-based matrix to c-TF-IDF """
        X = X * self._idf_diag
        X = normalize(X, axis=1, norm='l1', copy=False)
        return X
    
# from sklearn.datasets import fetch_20newsgroups
# newsgroups = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))

# Create documents per label
# docs = pd.DataFrame({'Document': newsgroups.data, 'Class': newsgroups.target})



def tfidf_exe(corpus,vocab=None):
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(corpus)

    npX = X.toarray()
    vals = npX.mean(axis=0).argsort()
    iv = {}
    for key,value in vectorizer.vocabulary_.items():
        iv[value]= key
#     print([iv[val] for val in vals[:20]])
#     print([iv[val] for val in vals[-20:]])
    return vectorizer


def extract_X(df,testdf,valdf,text_column = 'comment_text'):
    whp = WhiteSpacePreprocessing(df[text_column].values)
    pdc,upd,voc,rti = whp.preprocess()
    pdf = pd.DataFrame(df[text_column].values[rti],columns=['org_text'])
    pdf['class'] = df['class'].values[rti]
    pdf['post'] = pdf['org_text']
    pdf['pp'] = pdc
#     pdf['cls'] = df['cls'].values[rti]
#     rdf,c_vectorizer,X_c = ctfidf_exe(pdf)
    corpus = pdf.post.values
    t_vectorizer = tfidf_exe(corpus,vocab=voc)
    train_X = t_vectorizer.transform(df[text_column].values)
    test_X = t_vectorizer.transform(testdf[text_column].values)
    val_X = t_vectorizer.transform(valdf[text_column].values)
    
    
    return train_X, test_X, val_X

def train_test_split_df(data,train,test,val):
    data = data[data['class']!='explicit_hate']
    N = data.shape[0]
    train,test,val = int(N*train), int(N*test), int(N*val)
    train = N-(train+test+val)+train
    assert train+test+val == N
    shuffled_data = data.sample(frac=1.0).reset_index()
    cls = ['implicit_hate', 'not_hate']
    shuffled_data['targets'] = shuffled_data['class'].apply(lambda x: cls.index(x))
    train_data = shuffled_data.loc[:train-1]
    test_data = shuffled_data.loc[train:train+test]
    val_data = shuffled_data.loc[train+test:]
#     train_corpus, train_targets = train_data['post'].values, train_data['targets'].values
#     test_corpus, test_targets = test_data['post'].values, test_data['targets'].values
#     val_corpus, val_targets = val_data['post'].values, val_data['targets'].values
    
    return train_data, test_data, val_data

def random_masking(corpus,r=0.3):
    new_corpus = []
    for sent in corpus:
        new_corpus.append(' '.join([word if random.random()>r  else '[MASK]' for word in sent.split() ]))
    return new_corpus
# nc = random_masking(corpus)

# 2) tfidf 기반 마스킹
def ctfidf_exe(df,topn= 20):
    
    
    whp = WhiteSpacePreprocessing(df['post'].values)
    
    # preprocessed_docs, unpreprocessed_docs, vocabulary, retained_indices
    
    pdc,upd,voc,rti = whp.preprocess()
    
    pdf = pd.DataFrame(df['post'].values[rti],columns=['org_text'])
#     pdf['class'] = df['class'].values[rti]
    pdf['post'] = pdf['org_text']
    pdf['pp'] = pdc
    pdf['cls'] = df['cls'].values[rti]
    df = pdf
    
    cls = sorted(df.cls.unique())
    print("cls:",cls)
    
    docs = pd.DataFrame({"Document":df.pp.values, 'Class':df['cls'].values})
    docs_per_class = docs.groupby(['Class'], as_index=False).agg({'Document': ' '.join})
    # concat documents of each class (the number of rows is equal to the number of classes)

    count = CountVectorizer().fit_transform(docs_per_class.Document)
    
    c_vectorizer = CTFIDFVectorizer()
    ctfidf = c_vectorizer.fit_transform(count, n_samples=len(docs))
    # shouldn't it be equivalent to just TfidfVectorizer.fit_transform() ? hmm...
    # nope, it inherits from TfidfTransformer which inherits from TransformerMixin which provides a default fit_transform() which is just fit followed by transform
    
    count_vectorizer = CountVectorizer().fit(docs_per_class.Document)
    count = count_vectorizer.transform(docs_per_class.Document)
    
    words = count_vectorizer.get_feature_names()
    
    words_per_class = {cls[label]: [words[index] for index in ctfidf[label].toarray().argsort()[0][:topn]] 
                       for label in docs_per_class.Class}
    
    rdf = pd.DataFrame(words_per_class)
    
    return rdf,c_vectorizer,count_vectorizer,ctfidf

def tfidf_masking(corpus, bottom_n,top_n,tf_docs,ctfidf,r=0.8):
    bottom_words = [tf_docs[idx] for idx in ctfidf.argsort()[0][bottom_n:top_n]]
    
    # My guess about this function...
    # tf_docs is an array of words
    # ctfidf is an array with the 0th row representing the c-tf-idf values of each word for the 'offensive' class
    # and the 1st row representing the c-tf-idf values of each word for the 'non-offensive' class
    # ctfidf.argsort()[0][bottom_n:top_n] therefore gives the indices of words in tf_docs that are between bottom_n and top_n in terms of c-tf-idf values for the 'offensive' class
    # tf_docs and ctfidf are probably obtained through CTFIDFVectorizer
    
    bottom_words = set(bottom_words)
    new_corpus = []
    for sent in corpus:
        new_corpus.append(' '.join(['[MASK]' if (word.lower() in bottom_words) and (random.random()<r)  
                                    else word for word in sent.split() ]))
    return new_corpus

def tfidf_masking_one(corpus, bottom_n,top_n,tf_docs,ctfidf,r=0.8):
    
    bottom_words = [tf_docs[idx] for idx in ctfidf.argsort()[0][bottom_n:top_n]]
    bottom_words = set(bottom_words)
    new_corpus = []
    for sent in corpus:
        sent_ = []
        mask_idx = []
        for idx,word in enumerate(sent.split()):
            
            if (word.lower() in bottom_words) and (random.random()<r):
                mask_idx.append(idx)
            sent_.append(word)
        if len(mask_idx)!=0:
            idx = random.sample(mask_idx,k=1)[0]
            sent_[idx] = '[MASK]'
        new_corpus.append(' '.join(sent_))
    return new_corpus
def random_masking_one(corpus,r=0.3):
    new_corpus = []
    for sent in corpus:
        sent_ = []
        mask_idx = []
        for idx,word in enumerate(sent.split()):
            if (random.random()<r):
                mask_idx.append(idx)
            sent_.append(word)
        if len(mask_idx)!=0:
            idx = random.sample(mask_idx,k=1)[0]
            sent_[idx] = '[MASK]'
        new_corpus.append(' '.join(sent_))
    return new_corpus

def vocab_based_masking(corpus, vocab,r=0.8):
    vocab = set(vocab)
    new_corpus = []
    for sent in corpus:
        new_corpus.append(' '.join(['[MASK]' if (word in vocab) and (random.random()<r)  
                                    else word for word in sent.split() ]))
    return new_corpus