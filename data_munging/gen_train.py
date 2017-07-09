# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


# read data
print('Reading data...')
product_prior = pd.read_csv('../data/order_products__prior.csv')
product_train = pd.read_csv('../data/order_products__train.csv')
product_train = product_train[product_train.reordered==1]
orders = pd.read_csv('../data/orders.csv')
orders_train = orders[orders.eval_set=='train']
products = pd.read_csv('../data/products.csv')

print('Processing data...')
product_prior = product_prior.merge(orders[['order_id', 'user_id']], how='left',
                                    on='order_id')
product_train = product_train.merge(orders[['order_id', 'user_id']], how='left',
                                    on='order_id')

mergedata = product_prior.merge(product_train, how='left', on=['user_id', 'product_id'])

reordered = mergedata[~mergedata.reordered_y.isnull()]
reordered = reordered.drop(['order_id_y', 'add_to_cart_order_y', 'reordered_y'], axis=1)
reordered = reordered.rename(columns={
                                'order_id_x': 'order_id',
                                'add_to_cart_order_x': 'add_to_cart_order',
                                'reordered_x': 'reordered'
                            })
reordered_grouped = reordered.groupby(['user_id', 'product_id']) \
                             .agg({'add_to_cart_order': np.mean})
reordered_grouped = reordered_grouped.reset_index()
reordered_grouped = reordered_grouped.rename(
                        columns={'add_to_cart_order': 'add_to_cart_order_mean'})
reordered_grouped['label'] = 1

unreordered = mergedata[mergedata.reordered_y.isnull()]
unreordered = unreordered.drop(['order_id_y', 'add_to_cart_order_y', 'reordered_y'], axis=1)
unreordered = unreordered.rename(columns={'order_id_x': 'order_id', 
                            'add_to_cart_order_x': 'add_to_cart_order',
                            'reordered_x': 'reordered'
                            })
unreordered_grouped = unreordered.groupby(['user_id', 'product_id']) \
                                 .agg({'add_to_cart_order': np.mean})
unreordered_grouped = unreordered_grouped.reset_index()
unreordered_grouped = unreordered_grouped.rename(
                          columns={'add_to_cart_order': 'add_to_cart_order_mean'})
unreordered_grouped['label'] = 0

del mergedata
train_data = pd.concat([reordered_grouped, unreordered_grouped])
del reordered_grouped, unreordered_grouped
train_data = train_data.merge(orders_train, how='left', on='user_id')
#test_data = train_data[train_data.order_id.isnull()]
train_data = train_data[~train_data.order_id.isnull()]


# merge data
print('Merging data...')
#train_data = product_train.merge(orders_train, how='left', on='order_id')
train_data = train_data.merge(products, how='left', on='product_id')
train_data = pd.get_dummies(data=train_data, columns=['department_id', 'aisle_id'])

# save data
print('Saving data...')
cols_to_drop = ['order_id', 'product_id', 'user_id', 'eval_set', 'product_name']
train_data.drop(cols_to_drop, axis=1, inplace=True)
train_data.to_pickle('../data/train.pkl')

print('Done.')
