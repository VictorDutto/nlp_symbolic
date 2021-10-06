#!/usr/bin/env python
# coding: utf-8

# ## Installing fastText

# The first step of this tutorial is to install and build fastText.

# In[ ]:


get_ipython().system('wget https://github.com/facebookresearch/fastText/archive/v0.9.2.zip')
get_ipython().system('unzip v0.9.2.zip')
get_ipython().system('cd fastText-0.9.2 && make && pip install .')


# In[58]:


import fasttext


# ### Download directly with command line the english dataset

# In[6]:


get_ipython().system('cd fastText-0.9.2 && python3 download_model.py en')


# ### Import HuggingFace

# In[7]:


pip install datasets


# ### Loading the dataset "IMDB"

# In[19]:


from datasets import load_dataset


# Using the split argument, we can split the imdb into two separate dataset.

# In[83]:


train_dataset = load_dataset('imdb', split='train')
test_dataset = load_dataset('imdb', split='test')


# *train_dataset* is a class with two attributes :
# - Features which contains two features : text and label (our x_train and our y_train)
# - The number of rows in our dataset

# In[21]:


print(train_dataset)


# If we take the first sample of our training dataset, we can see the first review and the label (positive or negative) according to that review.

# In[22]:


print(train_dataset[0])


# ### Preprocessing the dataset

# With the review we previously saw, the review are definitely not preprocessed. There is still some html tags etc.. 
# We definitely to work on those.

# In[23]:


get_ipython().system('pip install contractions')
get_ipython().system('pip install beautifulsoup4')
get_ipython().system('pip install nltk')


# In[36]:


import contractions
from bs4 import BeautifulSoup
import unicodedata
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from nltk import WordPunctTokenizer
from nltk.stem import PorterStemmer
nltk.download('all')


import tqdm
from tqdm import tqdm


# In[ ]:


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


# In[ ]:


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


# In[ ]:


def expand_contractions(text: str) -> str:
    '''
    Expand contractions like "you're" to "you are".

            Parameters:
                    text (str): A text.

            Returns:
                    text (str): Text with no contractions.
    '''
    return contractions.fix(text)


# In[ ]:


def expand_contractions(text: str) -> str:
    '''
    Expand contractions like "you're" to "you are".

            Parameters:
                    text (str): A text.

            Returns:
                    text (str): Text with no contractions.
    '''
    return contractions.fix(text)


# In[ ]:


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


# In[ ]:


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


# In[ ]:


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


# In[47]:


en_stop = set(nltk.corpus.stopwords.words('english'))

def stop_words(text: str, stop_words: bool) -> str:
    '''
    ????

            Parameters:
                    text (str): A text.
                    stop_words (bool): True if you want the delete them, if not False.

            Returns:
                    text (str): Text without stop words.
    '''
    if (stop_words):
        tokens = text.split()
        tokens = [word for word in tokens if word not in en_stop]
        tokens = [word for word in tokens if len(word) > 3]
        preprocessed_text = ' '.join(tokens)
        return preprocessed_text
    
    return text


# In[118]:


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


pre_process_corpus = np.vectorize(pre_process_document)

x_train = []
x_test = []

for i in range(train_dataset.num_rows):
    x_train.append(pre_process_document(train_dataset[i]['text'], False, False, False))
    

for i in range(test_dataset.num_rows):
    x_test.append(pre_process_document(test_dataset[i]['text'], False, False, False))


# In[85]:


y_train = train_dataset['label']


# In[86]:


y_test = test_dataset['label']


# In[51]:


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


# In[52]:


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


# In[53]:


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


# In[54]:


create_imdb_train_file(train_dataset, y_train)
create_imdb_train_and_validation_file(train_dataset, y_train)
create_imdb_test_file(test_dataset, y_test)


# ## Let's train the model using fasttext.

# ##### 1) Lemming

# In[87]:


create_imdb_train_file(train_dataset, y_train, lemm = True)
create_imdb_train_and_validation_file(train_dataset, y_train, lemm = True)
create_imdb_test_file(test_dataset, y_test, lemm = True)


# In[88]:


get_ipython().run_cell_magic('time', '', 'model = fasttext.train_supervised(input="imdb_train_lemmed.txt")')


# In[91]:


res = model.test("imdb_test.txt")
print("Nombre de sample: " + str(res[0]) + ", Taux de précision: " + str(res[1]) + ", Rappel: " + str(res[2]))


# ##### 2) Stemming

# In[92]:


create_imdb_train_file(train_dataset, y_train, stemm = True)
create_imdb_train_and_validation_file(train_dataset, y_train, stemm = True)
create_imdb_test_file(test_dataset, y_test, stemm = True)


# In[93]:


get_ipython().run_cell_magic('time', '', 'model = fasttext.train_supervised(input="imdb_train_stemmed.txt")')


# In[94]:


res = model.test("imdb_test.txt")
print("Nombre de sample: " + str(res[0]) + ", Taux de précision: " + str(res[1]) + ", Rappel: " + str(res[2]))


# ##### 3) Stop Words

# In[95]:


create_imdb_train_file(train_dataset, y_train, stop_words = True)
create_imdb_train_and_validation_file(train_dataset, y_train, stop_words = True)
create_imdb_test_file(test_dataset, y_test, stop_words = True)


# In[96]:


get_ipython().run_cell_magic('time', '', 'model = fasttext.train_supervised(input="imdb_train_sw.txt")')


# In[97]:


res = model.test("imdb_test.txt")
print("Nombre de sample: " + str(res[0]) + ", Taux de précision: " + str(res[1]) + ", Rappel: " + str(res[2]))


# ##### 4) None

# In[98]:


get_ipython().run_cell_magic('time', '', 'model = fasttext.train_supervised(input="imdb_train.txt")')


# In[99]:


res = model.test("imdb_test.txt")
print("Nombre de sample: " + str(res[0]) + ", Taux de précision: " + str(res[1]) + ", Rappel: " + str(res[2]))


# ### Making the model better

# #### How about N-grams ?

# In[100]:


get_ipython().run_cell_magic('time', '', 'model = fasttext.train_supervised(input="imdb_train.txt", wordNgrams=2) ')


# In[101]:


res = model.test("imdb_test.txt")
print("Nombre de sample: " + str(res[0]) + ", Taux de précision: " + str(res[1]) + ", Rappel: " + str(res[2]))


# #### With more epochs and a smaller learning_rate ?

# In[102]:


get_ipython().run_cell_magic('time', '', 'model = fasttext.train_supervised(input="imdb_train.txt", lr=0.5, epoch=25, wordNgrams=2)')


# In[103]:


res = model.test("imdb_test.txt")
print("Nombre de sample: " + str(res[0]) + ", Taux de précision: " + str(res[1]) + ", Rappel: " + str(res[2]))


# #### With more epochs and a smaller smaller learning_rate ?

# In[104]:


get_ipython().run_cell_magic('time', '', 'model = fasttext.train_supervised(input="imdb_train.txt", lr=0.25, epoch=25, wordNgrams=2)')


# In[105]:


res = model.test("imdb_test.txt")
print("Nombre de sample: " + str(res[0]) + ", Taux de précision: " + str(res[1]) + ", Rappel: " + str(res[2]))


# #### With more epochs and a bigger learning_rate ?

# In[106]:


get_ipython().run_cell_magic('time', '', 'model = fasttext.train_supervised(input="imdb_train.txt", lr=1.0, epoch=25, wordNgrams=2)')


# In[107]:


res = model.test("imdb_test.txt")
print("Nombre de sample: " + str(res[0]) + ", Taux de précision: " + str(res[1]) + ", Rappel: " + str(res[2]))


# #### With the autotuneValidationfile (testing and performance purpose)

# In[108]:


get_ipython().run_cell_magic('time', '', 'model = fasttext.train_supervised(input="imdb_train_splited_with_the_validation.txt", autotuneValidationFile=\'imdb_validation.txt\', autotuneDuration=150)')


# In[109]:


res = model.test("imdb_test.txt")
print("Nombre de sample: " + str(res[0]) + ", Taux de précision: " + str(res[1]) + ", Rappel: " + str(res[2]))


# ## Beating the baseline

# ###### Glove

# In[110]:


get_ipython().system('pip install zeugma')


# In[119]:


from zeugma.embeddings import EmbeddingTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# In[120]:


glove = EmbeddingTransformer('glove')
x_train = glove.transform(x_train)


# In[121]:


model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(x_train, y_train)

x_test = glove.transform(x_test)
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))

