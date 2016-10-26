import os
import numpy as np
import csv
import time
import pandas as pd
from scipy import  stats
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import ensemble
from sklearn import svm
import datetime
map={0:0,1:1,2:1.95,3:2.90,4:3.85,5:4.80,6:5.75,7:6.70,8:7.}
train = pd.read_csv("learngood/farming.csv", dtype={'平均交易价格': pd.np.float64})
test = pd.read_csv("learngood/product_market.csv")
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
ra=0
rl=ra+1
rl=32476
for j in range(ra,rl):
 i=test.ix[j]
 market_name=i.市场名称映射值
 goods_name=i.农产品名称映射值
 if j !=ra and market_name == test.ix[j-1].市场名称映射值 and goods_name == test.ix[j-1].农产品名称映射值 :
     pass
 else :
     set=train[(train.市场名称映射值 == market_name) & (train.农产品名称映射值 == goods_name)]
     set = set.sort_values('数据发布时间')
     ttt=(set.shape[0])-1
     fix=set.iloc[ttt,9]
 # set=set.sort_values('数据发布时间')
 if set.shape[0]==0 :
     k.append(j)
     set=train[(train.农产品名称映射值 == goods_name)]
 pass
 if set.shape[0]==0:
  continue
 kk = np.ndarray((set.shape[0], 2))
 jj = 0
 for ii in set.数据发布时间.as_matrix():
   kk[jj, 0] = (datetime.datetime.strptime(ii, "%Y-%m-%d") - datetime.datetime(2006, 2, 25)).days % 365
   dp=((datetime.datetime.strptime(ii, "%Y-%m-%d") - datetime.datetime(2006, 2, 25)).days // 365)
   kk[jj,1]= dp-0.05*dp
   jj = jj + 1
 #model = linear_model.LinearRegression()
 if j !=ra and market_name == test.ix[j-1].市场名称映射值 and goods_name == test.ix[j-1].农产品名称映射值 :
     pass
 else:
     model = ensemble.GradientBoostingRegressor(loss='ls', learning_rate=0.01, n_estimators=1000, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=7, min_impurity_split=1e-07, init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')
     model.fit(kk, set.平均交易价格.as_matrix())
     model2=linear_model.Lasso()
     model2.fit(kk, set.平均交易价格.as_matrix())
     #params = {'n_estimators': 50, 'max_depth': 4, 'min_samples_split': 2,'learning_rate': 0.01, 'loss': 'ls'}
     #model=linear_model.LassoLarsCV()
     #model.fit(kk, set.平均交易价格.as_matrix())
 #dta = pd.Series(set.平均交易价格.as_matrix())
 #dta.index = pd.Index(set.数据发布时间.as_matrix())
 #dta.plot()
 #plt.show()
 pre= np.ndarray((1, 2))
 pre[0,0]=(datetime.datetime.strptime(test.ix[1].数据发布时间, "%Y-%m-%d") - datetime.datetime(2006, 2, 25)).days % 365
 dp=((datetime.datetime.strptime(test.ix[1].数据发布时间, "%Y-%m-%d") - datetime.datetime(2006, 2, 25)).days // 365)
 pre[0,1]=dp-0.05*dp
 #print(set)
 #print(pre[0,0])
 #print(pre[0,1])
 dd=model.predict(pre)
 test.loc[j,'单位']=dd[0]*0.7+fix*0.2
 #round(dd[0],2)
 print(test.loc[j,'单位'])
 print(j)
print(k)
test.to_csv("answersvr.csv", header=False ,index = False)
"""dta=pd.Series(set.平均交易价格.as_matrix())
dta.index=pd.Index(set.数据发布时间.as_matrix())
print(dta)"""
# for i in train[:]:
#6188
#15989
#21868