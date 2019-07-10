# Copyright (c) 2018 Yuichi Katori All Rights Reserved
# Author: Yuichi Katori (yuichi.katori@gmail.com)
# DONE: pandas groupbyによる平均・標準偏差の集計をエラーバーを含むグラフ

import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

names=['seed','id','NN','Nh','alpha_i','alpha_r','alpha_b','alpha_s','alpha0','alpha1','beta_i','beta_r','beta_b','Temp','lambda0','RMSE1','RMSE2','count_gap']
#df1=pd.read_csv("data_cbmrc6b_scan1_alpha_r.csv",sep=',',names=names)
#print(df1)

def plot1(filename,key):
    #key='alpha_r'
    df1=pd.read_csv(filename,sep=',',names=names)
    print(df1)
    df3 =df1.groupby(key).aggregate(['mean','std'])
    x1 = df1.groupby(key)[key].mean().values
    y1mean = df3['RMSE1','mean'].values
    y1std  = df3['RMSE1','std'].values
    y2mean = df3['count_gap','mean'].values
    y2std  = df3['count_gap','std'].values

    plt.subplot(2,1,1)
    plt.errorbar(x1,y1mean,yerr=y1std,fmt='o',color='b',ecolor='gray',capsize=2)
    plt.ylabel('RMSE1')

    plt.subplot(2,1,2)
    plt.errorbar(x1,y2mean,yerr=y2std,fmt='o',color='b',ecolor='gray',capsize=2)
    plt.ylabel('gap')

    plt.xlabel(key)
    plt.show()

#plot1("data_cbmrc6b_scan1_alpha_r.csv","alpha_r")
#plot1("data_cbmrc6b_scan1_alpha_b.csv","alpha_b")
#plot1("data_cbmrc6b_scan1_alpha_s.csv","alpha_s")

plot1("data_cbmrc6c_scan1_alpha_r.csv","alpha_r")
plot1("data_cbmrc6c_scan1_alpha_b.csv","alpha_b")
plot1("data_cbmrc6c_scan1_alpha_s.csv","alpha_s")
