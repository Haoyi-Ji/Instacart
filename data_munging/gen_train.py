# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


## read data
print('Reading data...')
product_prior = pd.read_csv('../data/order_products__prior.csv')
product_train = pd.read_csv('../data/order_products__train.csv')
product_train = product_train[product_train.reordered==1]
orders = pd.read_csv('../data/orders.csv')
orders_train = orders[orders.eval_set!='prior']
products = pd.read_csv('../data/products.csv')

print('Processing data...')
## labelling
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
del orders_train, label_data


## feature engineering
orders_prior = orders[orders.eval_set=='prior']
prior = orders_prior.merge(product_prior.drop('user_id', axis=1), 
                           how='left',
                           on='order_id').drop('eval_set', axis=1)

# user based features
usr = pd.DataFrame()
usr['usr_average_days_between_orders'] = prior.groupby('user_id')['days_since_prior_order'].mean().astype(np.float32)
usr['usr_n_orders'] = orders.groupby('user_id').size().astype(np.int16)
usr['usr_total_items'] = prior.groupby('user_id').size().astype(np.int16)
usr['usr_all_products'] = prior.groupby('user_id')['product_id'].apply(set)
usr['usr_total_distinct_items'] = (usr.usr_all_products.map(len)).astype(np.int16)
usr['usr_avg_basket'] = (usr.usr_total_items / usr.usr_n_orders).astype(np.float32)
usr = usr.drop(['usr_n_orders', 'usr_all_products'], axis=1).reset_index()
data = data.merge(usr, how='left', on='user_id')

# product based features

# user x product features


# static features
data = data.merge(products, how='left', on='product_id')
data = pd.get_dummies(data=data, columns=['department_id', 'aisle_id'])
cols_to_drop = ['user_id', 'product_name']
data.drop(cols_to_drop, axis=1, inplace=True)

## save data
print('Saving data...')
train_data = data[data.eval_set=='train'].drop(['product_id', 'order_id', 'eval_set'], axis=1)
test_data = data[data.eval_set=='test'].drop(['label', 'eval_set'], axis=1)
train_data.to_pickle('../data/train.pkl')
test_data.to_pickle('../data/test.pkl')

print('Done.')
