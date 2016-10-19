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
from sklearn import ensemble
from sklearn import svm
import datetime
train = pd.read_csv("农产品价格预测分析/farming.csv", dtype={'平均交易价格': pd.np.float64})
test = pd.read_csv("农产品价格预测分析/product_market.csv")
print(train.describe())
print(train.shape)
print(test.shape)
# train Columns: [农产品市场所在省份, 市场名称映射值, 农产品类别, 农产品名称映射值, 规格, 区域, 颜色, 单位, 最低交易价格, 平均交易价格, 最高交易价格, 数据入库时间, 数据发布时间]
# test  Columns: [农产品市场所在省份, 市场名称映射值, 农产品类别, 农产品名称映射值, 规格, 区域, 颜色, 单位, 数据入库时间, 数据发布时间]
test=test.loc[:,['市场名称映射值','农产品名称映射值','数据发布时间','单位']]
test['单位']=0
test.loc[1,'单位']=2222
print(test.head(5))
# print(train.head(1))
# print(test.head(1))
# unique_market=50
# unique_goods=2116
unique_market = train.市场名称映射值.unique()
unique_goods = train.农产品名称映射值.unique()
k=[]
for j in range(0,32476):
 i=test.ix[j]
 market_name=i.市场名称映射值
 goods_name=i.农产品名称映射值
 if j !=0 and market_name == test.ix[j-1].市场名称映射值 and goods_name == test.ix[j-1].农产品名称映射值 :
     pass
 else :
     set=train[(train.市场名称映射值 == market_name) & (train.农产品名称映射值 == goods_name)]
 # set=set.sort_values('数据发布时间')
 if set.shape[0]==0 :
     k.append(j)
     set=train[(train.农产品名称映射值 == goods_name)]
 pass
 if set.shape[0]==0:
  continue
 kk = np.ndarray((set.shape[0], 1))
 jj = 0
 for ii in set.数据发布时间.as_matrix():
   kk[jj, 0] = (datetime.datetime.strptime(ii, "%Y-%m-%d") - datetime.datetime(2006, 1, 1)).days
   jj = jj + 1
 #model = linear_model.LinearRegression()
 if j !=0 and market_name == test.ix[j-1].市场名称映射值 and goods_name == test.ix[j-1].农产品名称映射值 :
     pass
 else:
     #model = linear_model.ElasticNet()
     #model.fit(kk, set.平均交易价格.as_matrix())
     params = {'n_estimators': 50, 'max_depth': 4, 'min_samples_split': 2,'learning_rate': 0.01, 'loss': 'ls'}
     model=ensemble.GradientBoostingRegressor(**params)
     model.fit(kk, set.平均交易价格.as_matrix())
 test.loc[j,'单位']=model.predict((datetime.datetime.strptime(test.ix[1].数据发布时间, "%Y-%m-%d") - datetime.datetime(2006, 1, 1)).days )
 print(test.loc[j,'单位'])
 print(j)
print(k)
test.to_csv("answersvr.csv", header=False ,index = False)
"""dta=pd.Series(set.平均交易价格.as_matrix())
dta.index=pd.Index(set.数据发布时间.as_matrix())
print(dta)"""
# for i in train[:]:
