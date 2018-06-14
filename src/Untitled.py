
# coding: utf-8

# In[1]:

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


# In[2]:

train.head()

