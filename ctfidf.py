from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from nltk.corpus import stopwords as stop_words
import warnings
import numpy as np
import string
import scipy as sp
import pandas as pd

# test

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
        
        # filter only the words included in the top 2000 vocabulary & not in stopwords
        # & discard samples that don't include any of the filtered words
        # (why?) - my guess: discard too frequent / too rare words - why?

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
    
    
    

class CTfidfTransformer(TfidfTransformer):
    
    def __init__(self):
        super(CTfidfTransformer, self).__init__()
        
    def fit(self, X, n_samples, y=None):
        
        """
        X: sparse matrix of shape n_classes, n_features. A matrix of token counts.
        (can be obtained from get_class_count_matrix())
        n_samples: total number of samples (across all classes)
        """
        
        _, n_features = X.shape
        df = np.squeeze(np.asarray(X.sum(axis=0)))
        # document frequency: sum all count of terms across documents
        
        idf = np.log(n_samples / df)
        
        self._idf_diag = sp.sparse.diags(idf, offsets=0,
                                  shape=(n_features, n_features),
                                  format='csr',
                                  dtype=np.float64)
        
        return self
    
def get_class_count_matrix(df):
    
    """
    generate the count matrix X to feed the CTfidfTransformer
    
    df: preprocessed pandas dataframe of the dataset (should include the 'post' & 'targets' columns)
    can be obtained through dataset.load_dataframe() and dataset.extract_post_targets()
    """
    
    preprocessor = WhiteSpacePreprocessing(df['post'].values)
    
    preprocessed_docs, unpreprocessed_docs, vocabulary, retained_indices = preprocessor.preprocess()
    
    pdf = pd.DataFrame(df['post'].values[retained_indices], columns=['post'])
    pdf['preprocessed'] = preprocessed_docs
    pdf['targets'] = df['targets'].values[retained_indices]
    df = pdf
    
    n_docs = len(retained_indices) #FIXME should this be ALL documents instead of just retained ones?
    
    docs = pd.DataFrame({"Document":df.preprocessed.values, "Class":df.targets.values})
    docs_per_class = docs.groupby(['Class'], as_index=False).agg({'Document': ' '.join})

    count_vectorizer = CountVectorizer()
    count_matrix = count_vectorizer.fit_transform(docs_per_class.Document)
    
    feature_names = count_vectorizer.get_feature_names()
    
    return count_matrix, feature_names, n_docs
    
    
def get_ctfidf_matrix(df):
    
    count_matrix, feature_names, n_docs = get_class_count_matrix(df)
    
    ctfidf_transformer = CTfidfTransformer()
    
    return ctfidf_transformer.fit_transform(count_matrix, n_docs), feature_names
    
    
    
    
    
    
    
    