# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


# read data
print('Reading data...')
product_prior = pd.read_csv('../data/order_products__prior.csv')
product_train = pd.read_csv('../data/order_products__train.csv')
product_train = product_train[product_train.reordered==1]
orders = pd.read_csv('../data/orders.csv')
orders_train = orders[orders.eval_set!='prior']
products = pd.read_csv('../data/products.csv')

print('Processing data...')
# labelling
product_prior = product_prior.merge(orders[['order_id', 'user_id']], how='left',
                                    on='order_id')
product_train = product_train.merge(orders[['order_id', 'user_id']], how='left',
                                    on='order_id')

product_prior_record = product_prior[['user_id', 'product_id']].drop_duplicates()

label_data = product_prior_record.merge(product_train[['user_id', 'product_id', 'reordered']],
                                        how='left', on=['user_id', 'product_id'])

label_data['label'] = label_data.reordered.apply(lambda x: 1 if x==1 else 0)
label_data.drop('reordered', axis=1, inplace=True)

data = orders_train.merge(label_data, how='left', on='user_id')

# feature engineering
'''
add_to_cart_order = mergedata.groupby(['user_id', 'product_id']) \
                             .agg({'add_to_cart_order': np.mean}) \
                             .rename(columns={'add_to_cart_order': 'add_to_cart_order_mean'}) \
                             .reset_index()

mergedata = mergedata.merge(add_to_cart_order, how='left', on=['user_id', 'product_id'])

mergedata_grouped = mergedata.groupby(['user_id', 'product_id']) \
                             .agg({'add_to_cart_order': np.mean})
mergedata_grouped = mergedata_grouped.reset_index()
mergedata_grouped = mergedata_grouped.rename(
                        columns={'add_to_cart_order': 'add_to_cart_order_mean'})
'''

data = data.merge(products, how='left', on='product_id')
data = pd.get_dummies(data=data, columns=['department_id', 'aisle_id'])
cols_to_drop = ['product_id', 'user_id', 'product_name']
data.drop(cols_to_drop, axis=1, inplace=True)

# save data
print('Saving data...')
train_data = data[data.eval_set=='train'].drop(['order_id', 'eval_set'], axis=1)
test_data = data[data.eval_set=='test'].drop('eval_set', axis=1)
train_data.to_pickle('../data/train.pkl')
test_data.to_pickle('../data/test.pkl')

print('Done.')
