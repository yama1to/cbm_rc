# Copyright (c) 2018 Yuichi Katori All Rights Reserved
# Author: Yuichi Katori (yuichi.katori@gmail.com)
# DONE: pandas groupbyによる平均・標準偏差の集計をエラーバーを含むグラフ

import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

names=['seed','id','NN','Nh','alpha_i','alpha_r','alpha_b','alpha0','alpha1','alpha2','beta_i','beta_r','beta_b','Temp','lambda0','RMSE0','RMSE1','RMSE2','count_gap']
df1=pd.read_csv("data_cbmrc7b_opt1_rec.csv",sep=',',names=names)
print(df1)

def scatter():
    df2 = df1[  (df1['count_gap']<10) & (df1['RMSE2']<=0.1)]
    #df2 = df1[  (df1['count_gap']<1)]
    #df3 = df2[['alpha_r','alpha2','alpha_b','RMSE0','RMSE1','RMSE2','count_gap']]
    df3 = df2[['alpha_r','alpha2','alpha_b','RMSE0','RMSE1','RMSE2','count_gap']]
    #df2 = df1[['alpha_r','alpha2','alpha_b','RMSE0','RMSE1','RMSE2']]
    #df3 = df2
    #df3 = df2[  (df2['count_gap']<2)]
    #df3 = df2[ (df2['RMSE2']<=0.1) & (df2['count_gap']<100)]
    #df3 = df2[ (df2['RMSE2']<=0.1) & (df2['RMSE0']<=0.1) ]
    scatter_matrix(df3, alpha=0.8, figsize=(6, 6), diagonal='kde')
    plt.show()

#plot1()
scatter()
