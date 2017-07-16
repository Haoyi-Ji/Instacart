# -*- coding:utf-8 -*-
import pandas as pd
from gbdt_lr import GBDT_LR
from sklearn.externals import joblib

# read data and model
test_data = pd.read_pickle('../data/test.pkl')
model = joblib.load('gbdt_lr_model.pkl')

predictions = model.predict(test_data.drop(['order_id', 'product_id'], axis=1))
test_data['y_pred'] = predictions['y_pred']

res = test_data[['order_id', 'product_id', 'y_pred']]
res = res[res.y_pred==1].drop('y_pred', axis=1)
func = lambda x: ' '.join(x.product_id.astype(str).tolist())
submissions = res.groupby('order_id').apply(func).reset_index()
submissions = test_data.order_id.drop_duplicates().to_frame().merge(submissions,
                                                                    how='left',
                                                                    on='order_id')
submissions.rename(columns={0: 'products'}, inplace=True)
submissions.fillna('None', inplace=True)
submissions.sort_values('order_id', inplace=True)
submissions.to_csv('../data/submissions.csv', index=False)
