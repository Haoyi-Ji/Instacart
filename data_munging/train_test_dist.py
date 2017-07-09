# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

orders = pd.read_csv('../data/orders.csv')
orders[orders.eval_set=='train'].order_id.plot.hist(alpha=0.5, label='train', bins=30)
orders[orders.eval_set=='test'].order_id.plot.hist(alpha=0.5, label='test', bins=30)
orders[orders.eval_set=='prior'].order_id.plot.hist(alpha=0.5, label='prior', bins=30)
plt.title('train & test set distribition')
plt.xlabel('order_id')
plt.legend(loc='best')
plt.show()
#plt.savefig('../data/train_test_distribution.png')
