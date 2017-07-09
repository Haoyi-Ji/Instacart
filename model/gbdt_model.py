# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


print('Reading data...')
data = pd.read_pickle('../data/train.pkl')
y = data.label
X = data.drop(['label'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.25,
                                                    random_state=7)


print('Model fitting...')
gbdt = GradientBoostingClassifier()
gbdt.fit(X_train, y_train)
with open('gbdt_model.pkl', 'wb') as f:
    pickle.dump(gbdt, f)

#with open('gbdt_model.pkl', 'rb') as f:
#    gbdt = pickle.load(f)

y_pred = gbdt.predict(X_test)
print(len(y_pred))
print(len(set(y_pred)))
print(len(y_test))
F1Score = f1_score(y_test, y_pred)
print('F1 score: {}'.format(F1Score))
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: {}'.format(accuracy))
