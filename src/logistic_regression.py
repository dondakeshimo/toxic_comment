import numpy as np
import pandas as pd
import time
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack

import utils

start = time.time()

class_names = ["toxic",
               "severe_toxic",
               "obscene",
               "threat",
               "insult",
               "identity_hate"]

train = pd.read_csv("../data/train.csv").fillna(" ")
test = pd.read_csv("../data/test.csv").fillna(" ")

train = train[:1000]
test = test[:1000]

train_text = train["comment_text"]
test_text = test["comment_text"]
all_text = pd.concat([train_text, test_text])

elapsed = utils.show_elapsed_time("load data", start, 0)

word_vectorizer = TfidfVectorizer(sublinear_tf=True,
                                  strip_accents="unicode",
                                  analyzer="word",
                                  token_pattern=r"\w{1,}",
                                  stop_words="english",
                                  ngram_range=(1, 1),
                                  max_features=10000)
word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)
with open("../pickles/word_vectorizer.pickle", "wb") as f:
    pickle.dump(word_vectorizer, f)

elapsed = utils.show_elapsed_time("tf-idf word", start, elapsed)

char_vectorizer = TfidfVectorizer(sublinear_tf=True,
                                  strip_accents="unicode",
                                  analyzer="char",
                                  stop_words="english",
                                  ngram_range=(2, 6),
                                  max_features=50000)
char_vectorizer.fit(all_text)
train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)
with open("../pickles/char_vectorizer.pickle", "wb") as f:
    pickle.dump(char_vectorizer, f)

elapsed = utils.show_elapsed_time("tf-idf char", start, elapsed)

train_features = hstack([train_char_features, train_word_features])
test_features = hstack([test_char_features, test_word_features])

scores = []
submission = pd.DataFrame.from_dict({"id": test["id"]})
for class_name in class_names:
    train_target = train[class_name]
    classifier = LogisticRegression(solver="sag")

    cv_score = np.mean(cross_val_score(classifier,
                                       train_features,
                                       train_target,
                                       cv=3,
                                       scoring="roc_auc"))
    scores.append(cv_score)
    print("CV score for class {} is {}".format(class_name, cv_score))

    classifier.fit(train_features, train_target)
    submission[class_name] = classifier.predict_proba(test_features)[:, 1]
    with open("../pickles/{}.pickle".format(class_name), "wb") as f:
        pickle.dump(classifier, f)

    elapsed = utils.show_elapsed_time(class_name, start, elapsed)

print("Total CV score is {}".format(np.mean(scores)))

submission.to_csv("submission.csv", index=False)
elapsed = utils.show_elapsed_time("done", start, elapsed)
