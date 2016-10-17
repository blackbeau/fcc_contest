import os
import numpy as np
import csv
import time
import pandas as pd
from scipy import  stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
from sklearn import linear_model
import datetime
a=datetime.timedelta(100000)
print(a.days)
train = pd.read_csv("farmerpredict/farming.csv", dtype={'平均交易价格': pd.np.float64})
test = pd.read_csv("farmerpredict/product_market.csv")
print(train.shape)
print(test.shape)
# train Columns: [农产品市场所在省份, 市场名称映射值, 农产品类别, 农产品名称映射值, 规格, 区域, 颜色, 单位, 最低交易价格, 平均交易价格, 最高交易价格, 数据入库时间, 数据发布时间]
# test  Columns: [农产品市场所在省份, 市场名称映射值, 农产品类别, 农产品名称映射值, 规格, 区域, 颜色, 单位, 数据入库时间, 数据发布时间]
# print(train.head(1))
# print(test.head(1))
# unique_market=50
# unique_goods=2116
unique_market = train.市场名称映射值.unique()
unique_goods = train.农产品名称映射值.unique()
k=[]
for j in range(100,200):
 i=test.ix[j]
 market_name=i.市场名称映射值
 goods_name=i.农产品名称映射值
 set=train[(train.市场名称映射值 == market_name) & (train.农产品名称映射值 == goods_name)]
 # set=set.sort_values('数据发布时间')
 if set.shape[0]==0 :
     k.append(j)
     set=train[(train.农产品名称映射值 == goods_name)]
 pass
 if set.shape[0]==0:
  continue;
 kk = np.ndarray((set.shape[0], 1))
 jj = 0
 for ii in set.数据发布时间.as_matrix():
   kk[jj, 0] = (datetime.datetime.strptime(ii, "%Y-%m-%d") - datetime.datetime(2006, 1, 1)).days
   jj = jj + 1
 model = linear_model.LinearRegression()
 model.fit(kk, set.平均交易价格.as_matrix())
 print("ansssssssss")
 print(model.predict((datetime.datetime.strptime(test.ix[1].数据发布时间, "%Y-%m-%d") - datetime.datetime(2006, 1, 1)).days))
 print(j)
print(k)

"""dta=pd.Series(set.平均交易价格.as_matrix())
dta.index=pd.Index(set.数据发布时间.as_matrix())
print(dta)"""
# for i in train[:]:
