from typing import Callable
import numpy as np
import pandas as pd
import contractions
import unicodedata
import re

from tqdm import tqdm
from datasets import load_dataset, arrow_dataset
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression


def strip_html_tags(text : str) -> str:
    soup = BeautifulSoup(text, "html.parser")
    [s.extract() for s in soup(['iframe', 'script'])]
    stripped_text = soup.get_text()
    stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
    return stripped_text

def remove_accented_chars(text: str) -> str:
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def expand_contractions(text: str) -> str:
    return contractions.fix(text)

def remove_special_characters(text : str, remove_digits : bool=False) -> str:
    pattern = r'[^a-zA-Z0-9\s]' if not remove_digits else r'[^a-zA-Z\s]'
    text = re.sub(pattern, '', text)
    return text

def pre_process_document(document : str) -> str:
    """
    Preprocessing the given string, according to a set of rules
    
    Input: 
        A string
    
    Ouput:
        A string
    """
    # strip HTML
    document = strip_html_tags(document)
    # lower case
    document = document.lower()
    # remove extra newlines (often might be present in really noisy text)
    document = document.translate(document.maketrans("\n\t\r", "   "))
    # remove accented characters
    document = remove_accented_chars(document)
    # expand contractions    
    document = expand_contractions(document)  
    # remove special characters and\or digits    
    # insert spaces between special characters to isolate them    
    #special_char_pattern = re.compile(r'([{.(-)}])')
    #document = special_char_pattern.sub(" \\1 ", document)
    #document = remove_special_characters(document, remove_digits=True)  
    # remove extra whitespace
    document = re.sub(' +', ' ', document)
    document = document.strip()
    
    return document

def generate_x_and_y(dataset: arrow_dataset.Dataset) -> tuple:
    """
    From the dataset, returns the datas and their labels

    Input: 
        A dataset from dataset.load_dataset 

    Output: 
        A tuple of list
    """
    #creating the preprocessing which can be applied at once on the train dataset
    pre_process_corpus = np.vectorize(pre_process_document)

    #spliting x and y
    x_train = pre_process_corpus(dataset['text'])
    x_train_preprocessed = []

    for elt in x_train:
        x_train_preprocessed.append(np.array(elt.split(" ")))
    
    y_train = dataset['label']
    return (x_train_preprocessed, y_train)

def import_lexicon(path: str) -> pd.core.frame.DataFrame:
    """
    Loads a lexicon in a dataframe. Can return an error
    
    Input: 
        The path to a file. It must be a csv of at least 2 columns
    
    Output: 
        A dataframe
    """
    data = pd.read_csv(path, sep='\t', names=[0, 1, 2, 3])
    df = pd.DataFrame()
    df['token'] = data[0]
    df['sentiment'] = data[1]
    return df

def does_no_appear(review: np.str_) -> int:
    """
    Verify if the word no appears in the string given as argument
    
    Input:
        A string
    
    Ouput: 
        1 is no appears, else 0
    """
    if "no" in review:
        return 1
    return 0

def count_first_and_second_pro(review: np.str_) -> int:
    """
    Count the occurences of personnal pronouns in the string given as argument
    
    Input:
        A string
    
    Ouput: 
        The number of occurences of 1rst and 2nd person personnal pronouns
    """
    count = 0
    for word in review:
        if word in ["I", "i", "you", "yours"]:
            count += 1
    return count

def does_exclamation_appear(review: np.str_) -> int:
    """
    Verify if the word ! appears in the string given as argument
    
    Input: 
        A string
    
    Output:
        1 is ! appears, else 0
    """
    if "!" in review:
        return 1
    return 0

def log_word_count_in_doc(review: np.str_) -> int:
    """
    Compute the log of the number of words in the str given as argument
    
    Input: 
        A string
    
    Ouput:
        The number of occurences of 1rst and 2nd person personnal pronouns
    """
    return np.log(len(review))

def split_lexicon(lexicon: pd.core.frame.DataFrame) -> tuple:
    """
    Separate the lexicon in two parts
    
    Input:
        The lexicon
    
    Output:
        Two sub lexicons
    """
    return lexicon[lexicon.sentiment > 0], lexicon[lexicon.sentiment < 0]

def positivity_counter(review: np.str_ , positive_df: pd.core.frame.DataFrame) -> tuple:
    """
    Compute two features at once, the number of positive words and their sum
    
    Input: 
        The string to review and the positive lexicon
    
    Output: 
        The number of positive words and their sum
    """
    posi = np.isin(positive_df.token, review)
    #return the # of positive words and their sum
    return sum(posi), sum(positive_df.sentiment[posi])

def negativity_counter(review : np.str_, negative_df: pd.core.frame.DataFrame) -> tuple:
    """
    Compute two features at once, the number of negative words and their sum
    
    Input: 
        The string to review and the negative lexicon
    
    Output: 
        The number of negative words and their sum
    """
    nega = np.isin(negative_df.token, review)
    #return the # of negative words and their sum
    return sum(nega), sum(negative_df.sentiment[nega])

def LoRegression(x_train: list, y_train: list) -> np.ndarray:
    """
    Compute all the features for all the entries in x_train
    
    Input: 
        The data and their labels
    
    Ouput: 
        The array of all the features in the train set
    """
    lexicon = import_lexicon("vader_lexicon.txt")
    positive_df, negative_df = split_lexicon(lexicon)
    X_features = []
    for review in tqdm(x_train):
        feature = np.zeros(8)
        feature[0] = does_no_appear(review)
        feature[1] = does_exclamation_appear(review)
        feature[2] = count_first_and_second_pro(review)
        feature[3] = log_word_count_in_doc(review)
        feature[4], feature[5] = negativity_counter(review, negative_df)
        feature[6], feature[7] = positivity_counter(review, positive_df)
        X_features.append(feature)
    return np.asarray(X_features)