#!/usr/bin/env python
# coding: utf-8

get_ipython().system('wget https://github.com/facebookresearch/fastText/archive/v0.9.2.zip')
get_ipython().system('unzip v0.9.2.zip')
get_ipython().system('cd fastText-0.9.2 && make && pip install .')
get_ipython().system('cd fastText-0.9.2 && python3 download_model.py en')
get_ipython().system('pip install contractions')
get_ipython().system('pip install beautifulsoup4')
get_ipython().system('pip install nltk')
get_ipython().system('pip install datasets')


import fasttext
import contractions
import unicodedata
import numpy as np
import re
import nltk
import tqdm


from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from nltk import WordPunctTokenizer
from nltk.stem import PorterStemmer
from tqdm import tqdm
from datasets import load_dataset

nltk.download('all')

def strip_html_tags(text: str) -> str:
    '''
    Strip html tags from the input text.

            Parameters:
                    text (str): A text.

            Returns:
                    stripped_text (str): Text completely stripped without html tags.
    '''
    soup = BeautifulSoup(text, "html.parser")
    [s.extract() for s in soup(['iframe', 'script'])]
    stripped_text = soup.get_text()
    stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
    return stripped_text

def remove_accented_chars(text: str) -> str:
    '''
    Remove accented characters from the text.

            Parameters:
                    text (str): A text.

            Returns:
                    text (str): Text which does not contains accented characters.
    '''
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def expand_contractions(text: str) -> str:
    '''
    Expand contractions like "you're" to "you are".

            Parameters:
                    text (str): A text.

            Returns:
                    text (str): Text with no contractions.
    '''
    return contractions.fix(text)

def expand_contractions(text: str) -> str:
    '''
    Expand contractions like "you're" to "you are".

            Parameters:
                    text (str): A text.

            Returns:
                    text (str): Text with no contractions.
    '''
    return contractions.fix(text)

def remove_special_characters(text: str , remove_digits: bool) -> str:
    '''
    Remove special characters from the text.

            Parameters:
                    text (str): A text.
                    remove_digits (bool): True if you want to remove digits, else False.

            Returns:
                    text (str): Text which does not contains special characters (digits, etc...).
    '''
    pattern = r'[^a-zA-Z0-9\s]' if not remove_digits else r'[^a-zA-Z\s]'
    text = re.sub(pattern, '', text)
    return text

nltk.download('wordnet')

wordnet_lemmatizer = WordNetLemmatizer()

def lemming(text: str, lemming: bool) -> str:
    '''
    The words in our text input will be lemmatized to their root form. 
    Furthermore, the stop words and the words with the length less than 4 will be removed from the corpus.

            Parameters:
                    text (str): A text.
                    lemming (bool): True if you want the lemmitization, if not False.

            Returns:
                    text (str): Lemmed text.
    '''
    
    if (lemming):
        tokens = text.split()
        tokens = [wordnet_lemmatizer.lemmatize(word) for word in tokens]
        preprocessed_text = ' '.join(tokens)
        return preprocessed_text
    
    return text

ps = PorterStemmer()

def stemming(text: str, stemming: bool) -> str:
    '''
    Stemming is the process of producing morphological variants of a root/base word.
    We will use this function to do the same for our document.

            Parameters:
                    text (str): A text.
                    lemming (bool): True if you want the stemmatization, if not False.

            Returns:
                    text (str): Stemmed text.
    '''
    if (stemming):
        tokens = text.split()
        tokens = [ps.stem(word) for word in tokens]
        preprocessed_text = ' '.join(tokens)
        return preprocessed_text
    
    return text

def stop_words(text: str, stop_words: bool) -> str:
    '''
    Filter out the stop words within the text given as argument

            Parameters:
                    text (str): A text.
                    stop_words (bool): True if you want the delete them, if not False.

            Returns:
                    text (str): Text without stop words.
    '''
    en_stop = set(nltk.corpus.stopwords.words('english'))
    if (stop_words):
        tokens = text.split()
        tokens = [word for word in tokens if word not in en_stop]
        tokens = [word for word in tokens if len(word) > 3]
        preprocessed_text = ' '.join(tokens)
        return preprocessed_text
    
    return text

def pre_process_document(document: str, lemm: bool, stemm: bool, sw: bool) -> str:
    '''
    Preprocess the document to remove all the unnecessary information.

            Parameters:
                    text (str): A text.

            Returns:
                    text (str): Text pre-processed.
    '''
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
    special_char_pattern = re.compile(r'([{.(-)}])')
    document = special_char_pattern.sub(" \\1 ", document)
    document = remove_special_characters(document, True)  
    # remove extra whitespace
    document = re.sub(' +', ' ', document)
    document = document.strip()
    
    document = lemming(document, lemm)
    document = stemming(document, stemm)
    
    document = stop_words(document, sw)

    return document

def create_imdb_train_file(train_dataset, y_train: list,
                           lemm: bool = False, stemm:bool = False, stop_words:bool = False):
    '''
    Create the imdb train file according to the fasttext documentation.
    
             Parameters:
                     train_dataset: Training dataset
                     y_train : Label for the training dataset
    '''
    
    if (lemm):
        f = open("imdb_train_lemmed.txt", "w")
    
    elif (stemm):
        f = open("imdb_train_stemmed.txt", "w")
        
    elif (stop_words):
        f = open("imdb_train_sw.txt", "w")        

    else:
        f = open("imdb_train.txt", "w") 
    
    for i in tqdm(range(train_dataset.num_rows)):
        if y_train[i] == 1:
            f.write("__label__positive" + " " + pre_process_document(train_dataset[i]['text'], lemm, stemm, stop_words) + '\n')
        if y_train[i] == 0:
            f.write("__label__negative" + " " + pre_process_document(train_dataset[i]['text'], lemm, stemm, stop_words) + '\n')
    
    f.close()

def create_imdb_test_file(test_dataset, y_test: list, lemm: bool = False, stemm: bool = False, stop_words: bool = False):
    '''
    Create the imdb test file according to the fasttext documentation.
    
             Parameters:
                     test_dataset: Training dataset
                     y_test: Label for the testing dataset
    '''
    if (lemm):
        f = open("imdb_test_lemmed.txt", "w")
    
    elif (stemm):
        f = open("imdb_test_stemmed.txt", "w")

    elif (stop_words):
        f = open("imdb_test_sw.txt", "w")        

    else:
        f = open("imdb_test.txt", "w") 
    
    for i in tqdm(range(test_dataset.num_rows)):
        if y_test[i] == 1:
            f.write("__label__positive" + " " + pre_process_document(test_dataset[i]['text'], lemm, stemm, stop_words) + '\n')
        if y_test[i] == 0:
            f.write("__label__negative" + " " + pre_process_document(test_dataset[i]['text'], lemm, stemm, stop_words) + '\n')
            
    f.close()

def create_imdb_train_and_validation_file(train_dataset, y_train: list, lemm: bool = False,
                                          stemm: bool = False, stop_words: bool = False):
    '''
    Create the imdb validation file according to the fasttext documentation.
    
             Parameters:
                     train_dataset: Training dataset
                     y_train: Label for the training dataset
    '''
    i = 0
    
    if (lemm):
        f = open("imdb_validation_lemmed.txt", "w")
    
    elif (stemm):
        f = open("imdb_validation_stemmed.txt", "w")

    elif (stop_words):
        f = open("imdb_validation_sw.txt", "w")        

    else:
        f = open("imdb_validation.txt", "w") 
    
    for i in tqdm(range(5000)):
        if y_train[i] == 1:
            f.write("__label__positive" + " " + pre_process_document(train_dataset[i]['text'], lemm, stemm, stop_words) + '\n')
        if y_train[i] == 0:
            f.write("__label__negative" + " " + pre_process_document(train_dataset[i]['text'], lemm, stemm, stop_words) + '\n')
    
    f.close()
    
    if (lemm):
        f = open("imdb_train_splited_with_the_validation_lemmed.txt", "w")
    
    elif (stemm):
        f = open("imdb_train_splited_with_the_validation_stemmed.txt", "w")

    elif (stop_words):
        f = open("imdb_train_splited_with_the_validation_sw.txt", "w")        

    else:
        f = open("imdb_train_splited_with_the_validation.txt", "w") 
    
    while i != train_dataset.num_rows:
        if y_train[i] == 1:
            f.write("__label__positive" + " " + pre_process_document(train_dataset[i]['text'], lemm, stemm, stop_words) + '\n')
        if y_train[i] == 0:
            f.write("__label__negative" + " " + pre_process_document(train_dataset[i]['text'], lemm, stemm, stop_words) + '\n')
        i += 1
    
    f.close()

get_ipython().run_cell_magic('time', '', 'model = fasttext.train_supervised(input="imdb_train_lemmed.txt")')
get_ipython().run_cell_magic('time', '', 'model = fasttext.train_supervised(input="imdb_train.txt")')
get_ipython().run_cell_magic('time', '', 'model = fasttext.train_supervised(input="imdb_train.txt", wordNgrams=2) ')
get_ipython().run_cell_magic('time', '', 'model = fasttext.train_supervised(input="imdb_train.txt", lr=0.5, epoch=25, wordNgrams=2)')
get_ipython().run_cell_magic('time', '', 'model = fasttext.train_supervised(input="imdb_train.txt", lr=0.25, epoch=25, wordNgrams=2)')
get_ipython().run_cell_magic('time', '', 'model = fasttext.train_supervised(input="imdb_train.txt", lr=1.0, epoch=25, wordNgrams=2)')
get_ipython().run_cell_magic('time', '', 'model = fasttext.train_supervised(input="imdb_train_splited_with_the_validation.txt", autotuneValidationFile=\'imdb_validation.txt\', autotuneDuration=150)')
get_ipython().system('pip install zeugma')