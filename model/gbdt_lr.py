# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score

RANDOM_SEED = 7


class GBDT_LR(BaseEstimator, ClassifierMixin):
    def __init__(self, lr_penalty='l1', random_state=RANDOM_SEED):
        self.random_state = random_state
        self.grd = GradientBoostingClassifier()
        self.grd_enc = OneHotEncoder()
        self.lr = LogisticRegression(penalty=lr_penalty, random_state=random_state)
        self.threshold = 0


    def _calc_threshold(self, predicted_prob, y_true, lr_model, step=0.05, beta=1):
        predicted_prob = predicted_prob.T[lr_model.classes_.argmax()].T
        predicted_prob = pd.DataFrame(predicted_prob, columns=['proba'])
        predicted_prob['y_true'] = y_true.reset_index().drop('index', axis=1)

        threshold = step
        max_f_score = 0
        max_threshold = threshold
        while threshold < 1:
            print('Threshold = {}'.format(threshold))
            predicted_prob['y_pred'] = predicted_prob.proba.map(lambda x: 1 if x > threshold else 0)
            double_one_cnt = len(predicted_prob[predicted_prob.y_pred+predicted_prob.y_true==2])
            if double_one_cnt == 0:
                threshold += step
                continue
            p = double_one_cnt / predicted_prob.y_pred.sum()
            r = double_one_cnt / predicted_prob.y_true.sum()
            f_score = (1 + beta*beta)*p*r / (beta*beta*p + r)
            if f_score > max_f_score:
                max_f_score = f_score
                max_threshold = threshold
            threshold += step

        print('F_score: {}'.format(max_f_score))
        return max_threshold


    def fit(self, X_train, y_train):
        X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train, y_train,
                                                                    test_size=0.5,
                                                                    random_state=self.random_state)
        self.grd.fit(X_train, y_train)
        self.grd_enc.fit(self.grd.apply(X_train)[:, :, 0])
        self.lr.fit(self.grd_enc.transform(self.grd.apply(X_train_lr)[:, :, 0]), y_train_lr)
        predicted_prob = self.lr.predict_proba(self.grd_enc.transform(self.grd.apply(X_train_lr)[:, :, 0]))
        self.threshold = self._calc_threshold(predicted_prob, y_train_lr, self.lr)
        return self


    def predict(self, X_test):
        X_test = self.grd_enc.transform(self.grd.apply(X_test)[:, :, 0])
        predicted_prob = self.lr.predict_proba(X_test)
        predicted_prob = predicted_prob.T[self.lr.classes_.argmax()].T
        predicted_prob = pd.DataFrame(predicted_prob, columns=['proba'])
        predicted_prob['y_pred'] = predicted_prob.proba.apply(lambda x: 1 if x > self.threshold else 0)
        return predicted_prob



print('Reading data...')
data = pd.read_pickle('../data/train.pkl')
print('Sampling data...')
data = data.sample(200000, random_state=RANDOM_SEED)
y = data.label
X = data.drop(['label'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.25,
                                                    random_state=RANDOM_SEED)
clf = GBDT_LR()
print('Start training...')
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)
accuracy = accuracy_score(predicted['y_pred'], y_test)
print('Accuracy: {}'.format(accuracy))

# save model
with open('gbdt_lr_model.pkl', 'wb') as f:
    pickle.dump(clf, f)
