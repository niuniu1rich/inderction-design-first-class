import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('sale.csv')
df.drop_duplicates(subset=['ORDERID'],keep='first',inplace=True)
df.dtypes
df[df.isna().values == True]
df = df.dropna(how='any', axis
plt.figure(figsize=(10,5))
sns.distplot(df.AMOUNTINFO)
df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
df['Datediff'] = (pd.to_datetime('today') - df['ORDERDATE']).dt.days
df
R_Agg = df.groupby(by=['USERID'])['Datediff']
R_Agg = R_Agg.agg([('最近一次消费','min')])
F_Agg = df.groupby(by=['USERID'])['ORDERID']
F_Agg = F_Agg.agg([('2018年消费频次','count')])
M_Agg = df.groupby(by=['USERID'])
['AMOUNTINFO']
M_Agg = M_Agg.agg([('2018年消费金额',sum)])
rfm = R_Agg.join(F_Agg).join(M_Agg)
rfm
def rfm_convert (x):
  rfm_dict = (0:'R', 2:'M'}
  try:
    for i in range (0, 3, 2):
      bins = x. iloc[:, i]. quantile (q=np. linspace (0, 1, num=6) , interpolation=' nearest')
      if i == 0:
         labels = np. arange (5, 0, -1)
      else:
         labels = np. arange (1, 6)
      x[rfm_dict [ij] = pd. cut (x. iloc[:, il, bins=bins, labels=labels, include_lowest=True)
  except Exception as e:
  print (e)
rfm_convert（rfm)
bins = [1,3,5,12]
labels = np. arange (1, 4)
rfm['F']=pd.cut (rfm[2018年消费频次”]，bins=bins, labels=labels, include_lowest=True)
rfm. insert (4,'F',rfm.pop('F'))
rfm
rfm_model = rfm.filter(items=['R','F','M'])
rfm_model
def rfm(x):
    return x.i1oc[0]*3+x.i1oc[1]+x.i1oc[2]*3

rfm_model ['RFM'] = rfm_mode1.apply (rfm, axis=1)

bins = rfm_model.RFM.quantile (q=np. linspace (0, 1, num=9),interpolation='nearest')
labels =['流失客户','一般维持客户','新客户','潜力客户','重要挽留客户','重要深耕客户','重要唤回客户','重要价值客户']
rfm_model['Label of Customer'] = pd.cut (rfm_model.RFM, bins=bins, labels=1abe1s, include_1owest=True)
rfm_model
