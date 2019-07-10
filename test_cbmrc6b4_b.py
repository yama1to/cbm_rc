# Copyright (c) 2018 Katori lab. All Rights Reserved
# NOTE: plot 、作図用のスクリプト。ICJNNの論文原稿
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
exe = "python cbmrc6b4.py display=0 dataset=%d " % (dataset)
report = "data20190111_cbmrc6b4_ds%d.md" % (dataset)
prefix = "data20190111_cbmrc6b4_ds%d" % (dataset)

#report = "data20190110_cbmrc6b4_ds1.md"
prefix = "data20190110_cbmrc6b4_ds1"

text = "## %s  gridscan \n" % prefix
columns=['dataset','seed','id','NN','Nh','alpha_i','alpha_r','alpha_b','alpha_s','alpha0','alpha1','beta_i','beta_r','beta_b','Temp','lambda0','RMSE1','RMSE2','count_gap','overflow']

st.parallel=4
st.exe = exe
st.columns = columns
st.file_md = report
st.report(text)

def execute(): # RMSE1
    rows=4
    prefix1 = "data20190110_cbmrc6b4_ds1"
    prefix2 = "data20190110_cbmrc6b4_ds2"
    prefix3 = "data20190112_cbmrc6b4_ds3"
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
    #plt.legend()

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
    X1="alpha_s"
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
    X1="Temp"
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

def execute2(): # overflow
    rows=4
    prefix1 = "data20190110_cbmrc6b4_ds1"
    prefix2 = "data20190110_cbmrc6b4_ds2"
    prefix3 = "data20190112_cbmrc6b4_ds3"
    plt.figure(figsize=(8,24))
    #Y1="RMSE1"
    Y1="overflow"
    #(ymin,ymax)=(-0.02,0.8)
    plt.subplot(rows,1,1)
    X1="alpha_r"

    csv = prefix1+"_scan1ds_"+X1+".csv"
    df = pd.read_csv(csv,sep=',',names=columns)
    x,ymean,ystd,ymin,ymax = st.analyze(df,X1,Y1)
    plt.errorbar(x,ymean,yerr=ystd,fmt='o',color='b',ecolor='gray',capsize=2)
    plt.ylim(-0.02,0.8)
    plt.ylabel(Y1)
    plt.xlabel(X1)
    plt.legend()

    plt.subplot(rows,1,2)
    X1="alpha_i"
    csv = prefix1+"_scan1ds_"+X1+".csv"
    df = pd.read_csv(csv,sep=',',names=columns)
    x,ymean,ystd,ymin,ymax = st.analyze(df,X1,Y1)
    plt.errorbar(x,ymean,yerr=ystd,fmt='o',color='b',ecolor='gray',capsize=2)
    plt.ylim(-0.02,0.8)
    plt.ylabel(Y1)
    plt.xlabel(X1)

    plt.subplot(rows,1,3)
    X1="alpha_s"
    csv = prefix1+"_scan1ds_"+X1+".csv"
    df = pd.read_csv(csv,sep=',',names=columns)
    x,ymean,ystd,ymin,ymax = st.analyze(df,X1,Y1)
    plt.errorbar(x,ymean,yerr=ystd,fmt='o',color='b',ecolor='gray',capsize=2)
    plt.ylim(-0.02,0.8)
    plt.ylabel(Y1)
    plt.xlabel(X1)

    plt.subplot(rows,1,4)
    X1="Temp"
    csv = prefix1+"_scan1ds_"+X1+".csv"
    df = pd.read_csv(csv,sep=',',names=columns)
    x,ymean,ystd,ymin,ymax = st.analyze(df,X1,Y1)
    plt.errorbar(x,ymean,yerr=ystd,fmt='o',color='b',ecolor='gray',capsize=2)
    plt.ylim(-0.02,0.8)
    plt.ylabel(Y1)
    plt.xlabel(X1)

    plt.show()

#execute2()
