# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


print('Reading data...')
data = pd.read_pickle('../data/train.pkl')
y = data.label
X = data.drop(['label'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.25,
                                                    random_state=7)

'''
print('Model fitting...')
lr = LogisticRegression()
lr.fit(X_train, y_train)
with open('lr_model.pkl', 'wb') as f:
    pickle.dump(lr, f)
'''
with open('lr_model.pkl', 'rb') as f:
    lr = pickle.load(f)

y_pred = lr.predict(X_test)
print(len(y_pred))
print(len(set(y_pred)))
print(len(y_test))
F1Score = f1_score(y_test, y_pred)
print('F1 score: {}'.format(F1Score))
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: {}'.format(accuracy))
