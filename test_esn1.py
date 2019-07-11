# Copyright (c) 2018 Katori lab. All Rights Reserved
# NOTE:
import os
import datetime
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')## サーバ上で画像を出力するための設定。ローカルで表示する際はコメントアウト
import matplotlib.pyplot as plt
import optimization as opt
import searchtools as st

import sys
from arg2x import *

dataset=1
args = sys.argv
for s in args:
    dataset = arg2i(dataset,"dataset=",s)

### 共通設定
string_now = datetime.datetime.now().strftime("%Y%m%d") # 年月日
exe = "python esn1.py display=0 dataset=%d " % (dataset)
prefix = "data%s_esn1" % (string_now) # 実験名(data[日付]_[実験番号]_[モデル名])
path = prefix # 実験データを出力するディレクトリのパス
report = prefix + ".md" # マークダウンファイルを名前
columns=['dataset','seed','id','Nx','alpha_i','alpha_r','alpha_b','alpha0','tau','beta_i','beta_r','beta_b','lambda0','RMSE1','RMSE2']

st.set_path(path)
st.exe = exe
st.columns = columns
st.file_md = report
text="## esn1 (%s)  \n"  % (prefix)
st.report(text)

### 最適化
def func(row):
    return row['RMSE1'] + 0.1*row['count_gap']

def optimize():
    opt.exe=exe
    opt.set_path(path)
    opt.columns=columns
    opt.parallel=45
    opt.file_md = report
    opt.file_opt = prefix + "_opt.csv"

    opt.clear()
    opt.appendid()
    opt.appendseed()
    opt.append("alpha_r",value=0.8,min=0,max=1,round=2)
    opt.append("alpha_i",value=0.8,min=0,max=2,round=2)
    opt.append("alpha0" ,value=0.7,min=0,max=2,round=2)
    opt.append("tau"    ,value=2.0,min=0,max=5,round=2)

    opt.config()
    opt.minimize({'target':"RMSE1",'iteration':10,'population':10,'samples':10})
    #opt.minimize({'TARGET':func,'iteration':10,'population':20,'samples':20})
#optimize()

### グリッドサーチ
st.exe = exe

#png = prefix+"_test.png"
#st.execute1(exe,file_fig1=png)

def gridsearch(X1,min=0,max=1,num=41,samples=10):
    csv = prefix+"_scan1ds_"+X1+".csv"
    png = prefix+"_scan1ds_"+X1+".png"

    print("scaning...")
    st.scan1ds(csv,X1,min=min,max=max,num=num,samples=samples,run=1,list=0)

    #df = pd.read_csv(csv,sep=',',names=columns)
    df = st.load_dataframe(csv)

    print("ploting...")
    plt.figure()

    plt.subplot(2,1,1)
    x,ymean,ystd,ymin,ymax = st.analyze(df,X1,"RMSE1")
    plt.errorbar(x,ymean,yerr=ystd,fmt='o',color='b',ecolor='gray',capsize=2)
    plt.ylabel('RMSE1')

    #plt.subplot(2,1,2)
    #x,ymean,ystd,ymin,ymax = st.analyze(df,X1,"count_gap")
    #plt.errorbar(x,ymean,yerr=ystd,fmt='o',color='b',ecolor='gray',capsize=2)
    #plt.ylabel('gap')

    plt.xlabel(X1)

    st.plt_output(png)

#gridsearch("alpha_r",min=0,max=2,num=51,samples=20)
#gridsearch("alpha_i",min=0,max=1,num=51,samples=20)
#gridsearch("alpha0" ,min=0,max=1,num=51,samples=20)
gridsearch("tau"    ,min=1,max=10,num=11,samples=2)
#gridsearch("beta_i" ,min=0,max=1,num=51,samples=20)
#gridsearch("beta_r" ,min=0,max=1,num=51,samples=20)
"""
alpha_r = 0.8
alpha_b = 0.0 # 0.8
alpha_i = 0.8
beta_r = 0.1
beta_b = 0.1
beta_i = 0.1
alpha0 = 0.7
tau = 2
"""
