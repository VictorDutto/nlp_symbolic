from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow
from tensorflow import keras
from sklearn.metrics import classification_report


(x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.imdb.load_data(
    path="imdb.npz",
    num_words=None,
    skip_top=0,
    maxlen=None,
    seed=113,
    start_char=1,
    oov_char=2,
    index_from=3
)

def import_lexicon(path):
    data = pd.read_csv(path, sep='\t', names=[0, 1, 2, 3])
    df = pd.DataFrame()
    df['token'] = data[0]
    df['sentiment'] = data[1]
    return df

def get_reviews_from_class(x_train, y_train, _class, V):
    reviews = x_train[y_train == _class]
    
    for i in range (len(reviews)):
        decode_sentence = reviews[i]
        reviews[i] = decode_sentence
    return reviews

def does_no_appear(review) -> int:
    if "no" in review:
        return 1
    return 0


def count_first_and_second_pro(review) -> int:
    count = 0
    for char in review:
        if char in ["I", "i", "you", "yours"]:
            count += 1
    return count

def exclamation_in_doc(review):
    if "!" in review:
            return 1
    return 0

def log_word_count_in_doc(review):
    return np.log(len(review))

def number_of_words_pos(review, lexicon):
    columns = ['token']
    tmp = lexicon[lexicon.sentiment > 0]
    positive_words = tmp[columns].to_numpy().tolist()
    number_of_pos = np.in1d(positive_words, review)
    return sum(number_of_pos)

def number_of_words_neg(review, lexicon):
    columns = ['token']
    tmp = lexicon[lexicon.sentiment < 0]
    negative_words = tmp[columns].to_numpy().tolist()
    number_of_neg = np.in1d(negative_words, review)
    return sum(number_of_neg)

def LoRegression(X_train, y_train):
    nb_class = 2
    lexicon = import_lexicon("vader_lexicon.txt")
    X_features_of_all_the_class = []
    V = keras.datasets.imdb.get_word_index()

    # Preprocessing

    for _class in range (nb_class):
        reviews = get_reviews_from_class(X_train, y_train, _class, V)
        features = []

        for review in tqdm(reviews):
            feature = []
            feature.append(does_no_appear(review))
            feature.append(count_first_and_second_pro(review))
            feature.append(exclamation_in_doc(review))
            feature.append(log_word_count_in_doc(review))
            feature.append(number_of_words_neg(review, lexicon))
            feature.append(number_of_words_pos(review, lexicon))
            features.append(feature)

        X_features_of_all_the_class.append(features)
    return np.asarray(X_features_of_all_the_class)


X_features_of_all_the_class = LoRegression(x_train, y_train)

X_features = X_features_of_all_the_class.reshape(X_features_of_all_the_class.shape[0] * X_features_of_all_the_class.shape[1], X_features_of_all_the_class.shape[2])

clf = LogisticRegression(random_state=0).fit(X_features_of_all_the_class, y_train)


#y_test = np.resize(y_test, 150000).reshape(25000,6)
y_test = y_test.reshape(25000, 6)
print(y_test.shape)
y_pred = clf.predict(y_test)
target_names = ['class 0', 'class 1']
print(classification_report(y_test, y_pred, target_names=target_names))