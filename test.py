import os
import numpy as np
import csv
import pandas as pd

train=pd.read_csv("farmerpredict/farming.csv",dtype={'平均交易价格': pd.np.float64})
test=pd.read_csv("farmerpredict/product_market.csv")
print(train.shape)
print(test.shape)
#train Columns: [农产品市场所在省份, 市场名称映射值, 农产品类别, 农产品名称映射值, 规格, 区域, 颜色, 单位, 最低交易价格, 平均交易价格, 最高交易价格, 数据入库时间, 数据发布时间]
#test  Columns: [农产品市场所在省份, 市场名称映射值, 农产品类别, 农产品名称映射值, 规格, 区域, 颜色, 单位, 数据入库时间, 数据发布时间]
#print(train.head(1))
#print(test.head(1))
print(np.NAN)
#unique_market=50
#unique_goods=2116
unique_market=train.市场名称映射值.unique()
unique_goods=train.农产品名称映射值.unique()
current_data=[]
i=train[0:1]



#for i in train[:]:



