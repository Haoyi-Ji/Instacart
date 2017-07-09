# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression





predicted_prob = lr.predict_prob(X_test)


def calc_threshold(predicted_prob, y_true, step=0.05, beta=1):
    predicted_prob = predicted_prob.T[lr.classes_.argmax()].T
    predicted_prob = pd.DataFrame(predicted_prob, columns=['proba'])
    predicted_prob['y_true'] = y_true.reset_index().drop('index', axis=1)

    threshold = step
    max_f_score = 0
    max_threshold = threshold
    while threshold < 1:
        predicted_prob['y_pred'] = predicted_prob.proba.apply(lambda x: 1 if x > threshold else 0)
        double_one_cnt = len(predicted_prob[predicted_prob.y_pred+predicted_prob.y_true==2])
        if double_one_cnt == 0:
            continue
        p = double_one_cnt / predicted_prob.y_pred.sum()
        r = double_one_cnt / predicted_prob.y_true.sum()
        f_score = (1 + beta*beta)*p*r / (beta*beta*p + r)
        if f_score > max_f_score:
            max_f_score = f_score
            max_threshold = threshold
        threshold += step

    return max_f_score, max_threshold
