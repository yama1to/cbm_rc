# Copyright (c) 2018 Katori lab. All Rights Reserved
# NOTE: plot、作図用
import numpy as np
import pandas as pd
import matplotlib as mpl
#mpl.use('Agg')## サーバ上で画像を出力するための設定。ローカルで表示する際はコメントアウト
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
exe = "python esn1.py display=0 dataset=%d " % (dataset)
report = "data20190113_esn1.md"
prefix = "data20190113_esn1_ds%d" % (dataset)
columns=['dataset','seed','id','Nx','alpha_i','alpha_r','alpha_b','alpha0','tau','beta_i','beta_r','beta_b','lambda0','RMSE1','RMSE2']

#st.exe = exe
#st.columns = columns
#st.file_md = report
#text="## esn1 (%s)  \n"  % (prefix)
#st.report(text)

### グリッドサーチ
st.exe = exe
def execute(): # RMSE1
    rows=4
    prefix1 = "data20190113_esn1_ds1"
    prefix2 = "data20190113_esn1_ds2"
    prefix3 = "data20190113_esn1_ds3"
    plt.figure(figsize=(8,24))
    Y1="RMSE1"
    #Y1="overflow"

    plt.subplot(rows,1,1)
    X1="alpha_r"

    csv = prefix1+"_scan1ds_"+X1+".csv"
    df = pd.read_csv(csv,sep=',',names=columns)
    x,ymean,ystd,ymin,ymax = st.analyze(df,X1,Y1)
    plt.errorbar(x,ymean,yerr=ystd,fmt='o',color='b',ecolor='gray',capsize=2,label="s.s")

    csv = prefix2+"_scan1ds_"+X1+".csv"
    df = pd.read_csv(csv,sep=',',names=columns)
    x,ymean,ystd,ymin,ymax = st.analyze(df,X1,Y1)
    plt.errorbar(x,ymean,yerr=ystd,fmt='o',color='r',ecolor='gray',capsize=2,label="c.s")

    csv = prefix3+"_scan1ds_"+X1+".csv"
    df = pd.read_csv(csv,sep=',',names=columns)
    x,ymean,ystd,ymin,ymax = st.analyze(df,X1,Y1)
    plt.errorbar(x,ymean,yerr=ystd,fmt='o',color='g',ecolor='gray',capsize=2,label="chaos")

    plt.ylim(-0.0,1)
    plt.ylabel(Y1)
    plt.xlabel(X1)
    plt.legend()

    plt.subplot(rows,1,2)
    X1="alpha_i"

    csv = prefix1+"_scan1ds_"+X1+".csv"
    df = pd.read_csv(csv,sep=',',names=columns)
    x,ymean,ystd,ymin,ymax = st.analyze(df,X1,Y1)
    plt.errorbar(x,ymean,yerr=ystd,fmt='o',color='b',ecolor='gray',capsize=2)
    csv = prefix2+"_scan1ds_"+X1+".csv"
    df = pd.read_csv(csv,sep=',',names=columns)
    x,ymean,ystd,ymin,ymax = st.analyze(df,X1,Y1)
    plt.errorbar(x,ymean,yerr=ystd,fmt='o',color='r',ecolor='gray',capsize=2)
    csv = prefix3+"_scan1ds_"+X1+".csv"
    df = pd.read_csv(csv,sep=',',names=columns)
    x,ymean,ystd,ymin,ymax = st.analyze(df,X1,Y1)
    plt.errorbar(x,ymean,yerr=ystd,fmt='o',color='g',ecolor='gray',capsize=2)
    plt.ylim(-0.0,1)
    plt.ylabel(Y1)
    plt.xlabel(X1)

    plt.subplot(rows,1,3)
    X1="alpha0"
    csv = prefix1+"_scan1ds_"+X1+".csv"
    df = pd.read_csv(csv,sep=',',names=columns)
    x,ymean,ystd,ymin,ymax = st.analyze(df,X1,Y1)
    plt.errorbar(x,ymean,yerr=ystd,fmt='o',color='b',ecolor='gray',capsize=2)
    csv = prefix2+"_scan1ds_"+X1+".csv"
    df = pd.read_csv(csv,sep=',',names=columns)
    x,ymean,ystd,ymin,ymax = st.analyze(df,X1,Y1)
    plt.errorbar(x,ymean,yerr=ystd,fmt='o',color='r',ecolor='gray',capsize=2)
    csv = prefix3+"_scan1ds_"+X1+".csv"
    df = pd.read_csv(csv,sep=',',names=columns)
    x,ymean,ystd,ymin,ymax = st.analyze(df,X1,Y1)
    plt.errorbar(x,ymean,yerr=ystd,fmt='o',color='g',ecolor='gray',capsize=2)
    plt.ylim(-0.0,1)
    plt.ylabel(Y1)
    plt.xlabel(X1)

    plt.subplot(rows,1,4)
    X1="tau"
    csv = prefix1+"_scan1ds_"+X1+".csv"
    df = pd.read_csv(csv,sep=',',names=columns)
    x,ymean,ystd,ymin,ymax = st.analyze(df,X1,Y1)
    plt.errorbar(x,ymean,yerr=ystd,fmt='o',color='b',ecolor='gray',capsize=2)
    csv = prefix2+"_scan1ds_"+X1+".csv"
    df = pd.read_csv(csv,sep=',',names=columns)
    x,ymean,ystd,ymin,ymax = st.analyze(df,X1,Y1)
    plt.errorbar(x,ymean,yerr=ystd,fmt='o',color='r',ecolor='gray',capsize=2)
    csv = prefix3+"_scan1ds_"+X1+".csv"
    df = pd.read_csv(csv,sep=',',names=columns)
    x,ymean,ystd,ymin,ymax = st.analyze(df,X1,Y1)
    plt.errorbar(x,ymean,yerr=ystd,fmt='o',color='g',ecolor='gray',capsize=2)

    plt.ylim(-0.0,1.)
    plt.ylabel(Y1)
    plt.xlabel(X1)

    plt.show()
execute()
