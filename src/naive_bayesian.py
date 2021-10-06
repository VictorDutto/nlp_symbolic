import tensorflow
from tensorflow import keras

from tqdm import tqdm
import numpy as np
import math
import pandas as pd
import re

from datasets import load_dataset
from sklearn.metrics import classification_report, precision_recall_fscore_support
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize 

# Importing IMDB Dataset and split in train and test

# Creates a dictionary of words with the index of the word as value and the keyword itself as key
word_index = keras.datasets.imdb.get_word_index()

# Creates a dictionary of words with the index in the IMDB dataset as key
inverted_word_index = dict([(i, word) for (word, i) in word_index.items()])

def decode_sentence(review: list, string: bool = True, offset: int = 3) -> str:
    """
    Decode a sentence from the IMDB dataset
    
    Inputs: The document to decode
            A Boolean to put the result in string or in list
            An offset to decode the review properly
    
    Output: The decoded review 
    """
    if string:
        return " ".join(inverted_word_index.get(i - offset, "?") for i in review)
    
    return [inverted_word_index.get(i - offset, "?")  for i in review]

def decode_all_review(x_train: list, string: bool = True, offset: int = 3) -> list:
    """
    Decode all the review from a list of review
    
    Inputs: The list of review to decode
            A Boolean to put the result in string or in list
            An offset to decode the review properly
    
    Output: The filtered list of review 
    """
    reviews = []
    
    for i in range (len(x_train)):
        reviews.append(decode_sentence(x_train[i], string, offset))

    return reviews

def get_total_length(x_train: list) -> int:
    """
    Get the number of words in an array of review
    
    Inputs: The array of review
            
    Output: The number of words in the array 
    """
    res = 0
    for l in x_train:
        res += len(l)
    return res

def naive_bayes_classifier_train(x_train: list, y_train: list) -> tuple:
    """
    Naive bayes Classifier training function

    Args:
        x train being the numerical values of words in the imdb dataset
        and y train its respective class, also noted as a numerical value

    Output: 
        An array of class respective weights, 
        A dict of probability for the key to be in each class, 
        The vocabulary
    """
    
    # tuple value/label in each element from imdb
    ndoc = len(x_train)
    nbclasses = len(np.unique(y_train))
    logprior = [0] * nbclasses
    bigdoc = [[]] * nbclasses
    voca = list(keras.datasets.imdb.get_word_index().values())
    loglikelyhood = np.ndarray((len(voca), nbclasses))
    
    for c in range(nbclasses):
        nlogdoc = math.log(ndoc)
        # masking the array to filter it
        c_class_train = x_train[y_train == c]
        nc = len(c_class_train)
        logprior[c] =  math.log(nc) / nlogdoc
        # getting the number of occurences in the vocabulary
        occu_voca = np.zeros(len(voca) + 1)
        
        for docu in tqdm(c_class_train):
            docu_hist = np.histogram(docu, bins=np.arange(len(voca) + 2))[0]
            occu_voca = np.add(occu_voca, docu_hist)
        
        nb_word_total  = get_total_length(c_class_train)
        
        for index in tqdm(range(len(voca))):
            value = math.log((occu_voca[index] + 1) / (nb_word_total + 1))
            loglikelyhood[index, c] = value
    
    return logprior, loglikelyhood, voca

def test_naive_bayes(testdoc: list, logprior: list, loglikelyhood: np.ndarray, C: list, V: dict) -> np.int64:
    """
    Naive bayes Classifier test function
    
    Inputs: The document on which we test
            The logprior vector
            The loglikelyhood vector
            The classes
            The vocabulary
    
    Output: The document class 
    """

    _sum = [0] * len(C)
    
    for c in range(len(C)):
        _sum[c] = logprior[c]
    
        for word in testdoc:
            try:
                #voca starts with a minimal value of 1
                #thus we match voca and loglikelyhood
                _sum[c] += loglikelyhood[word, c]
    
            except IndexError:
                #unrecognized word, thus we are not knowing in which class it belongs
                pass
    
    return np.argmax(_sum)

def predict(x_test: list, logprior: np.ndarray, loglikelyhood: np.ndarray) -> np.ndarray:
    """
    Predict the class of an array of reviews
    
    Inputs: The array of reviews
    
    Output: The array of predicted class 
    """
    
    test_card = x_test.shape[0]
    res = np.zeros(test_card)
    
    for index in range(test_card):
        y_pred = test_naive_bayes(x_test[index], logprior, loglikelyhood, [0, 1], word_index)
        res[index] = y_pred

    return res

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def encode_set_of_words(set_words: set) -> list:
    """
    Encode a set of words

    Inputs: The set of words to encode
    
    Output: The encoded set of words
    """
    L = []

    for word in set_words:
        try:
            L.append(word_index[word])

        except KeyError:
            pass

    return L

stop_words_encoded = encode_set_of_words(stop_words)

def delete_stop_words(document: list) -> list:
    """
    Delete stop words function
    
    Inputs: The document to filter
    
    Output: The filtered document 
    """
    delete_res = document.copy()
    
    for stop_word in stop_words_encoded:
        try :
            while (True):
                delete_res.remove(stop_word)
        
        except ValueError:
            pass
    
    return delete_res

def stemming(document: list, re_upper: re.Pattern, stemmer: nltk.stem.snowball.SnowballStemmer) -> list:
    """
    Function of Stemming
    
    Inputs: The document to stemmed
            A regex that matches all the words(alphabetical)
            The Stemmer
    
    Output: The stemmed document 
    """
    stemmed = [stemmer.stem(word) for word in document if re_upper.match(word)]
    
    return encode_set_of_words(stemmed)

nltk.download('wordnet')
wordnet_lemmatizer = WordNetLemmatizer()

def lemming(document: list) -> list:
    """
    Function of Lemming
    
    Inputs: The document to lemmed
    
    Output: The lemmed document 
    """
    lemmas = [wordnet_lemmatizer.lemmatize(token) for token in document]
    
    return encode_set_of_words(lemmas)

def binary_naive_bayes_classifier_train(x_train: list, y_train: list) -> tuple:
    """
    Binary Naive bayes Classifier training function

    Args:
        x train being the numerical values of words in the imdb dataset
        and y train its respective class, also noted as a numerical value

    Output: 
        An array of class respective weights, 
        A dict of probability for the key to be in each class, 
        The vocabulary
    """
    
    # tuple value/label in each element from imdb
    ndoc = len(x_train)
    nbclasses = len(np.unique(y_train))
    logprior = [0] * nbclasses
    bigdoc = [[]] * nbclasses
    voca = list(keras.datasets.imdb.get_word_index().values())
    presence_matrix = np.ndarray((len(voca), nbclasses))
    arr_voc = np.arange(len(voca) + 1)
    
    for c in range(nbclasses):
        nlogdoc = math.log(ndoc)
        # masking the array to filter it
        c_class_train = x_train[y_train == c]
        nc = len(c_class_train)
        logprior[c] =  math.log(nc) / nlogdoc
        # getting the number of occurences in the vocabulary
        voca_acc = np.zeros(len(voca) + 1)
    
        for docu in tqdm(c_class_train):
            docu_hist = np.isin(arr_voc, docu)
            voca_acc += docu_hist
    
        for index in tqdm(range(len(voca))):
            if voca_acc[index] > 0:
                presence_matrix[index, c] = 1
            else:
                presence_matrix[index, c] = 0
    
    return logprior, presence_matrix, voca