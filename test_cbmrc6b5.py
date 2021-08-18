# Copyright (c) 2018 Katori lab. All Rights Reserved
# NOTE:
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')## サーバ上で画像を出力するための設定。ローカルで表示する際はコメントアウト
import matplotlib.pyplot as plt
import optimization as opt
import searchtools as st

import sys
from arg2x import *


dataset=3
args = sys.argv
for s in args:
    dataset = arg2i(dataset,"dataset=",s)

### 共通設定
exe = "python cbmrc6b4tmp.py display=0 dataset=%d " % (dataset)
report = "data20190113_cbmrc6b4_ds%d.md" % (dataset)
prefix = "data20190113_cbmrc6b4_ds%d" % (dataset)

#report = "data20190110_cbmrc6b4_ds1.md"
#prefix = "data20190110_cbmrc6b4_ds1"

text = "## %s  gridscan \n" % prefix

columns=['dataset','seed','id','NN','Nh','alpha_i','alpha_r','alpha_b','alpha_s','alpha0','alpha1','beta_i','beta_r','beta_b','Temp','lambda0','RMSE1','RMSE2','count_gap','overflow']

st.parallel=4
st.exe = exe
st.columns = columns
st.file_md = report
st.report(text)

### 最適化
def func(row):
    return row['RMSE1'] + 0.1*row['count_gap']

def optimize():
    opt.exe=exe
    opt.columns=columns
    opt.parallel=45
    opt.file_md = report
    opt.file_opt = prefix + "_opt.csv"

    opt.clear()
    opt.appendid()
    opt.appendseed()
    opt.append("alpha_r",value=0.2,min=0,max=1,round=2)
    opt.append("alpha_i",value=0.2,min=0,max=1,round=2)
    opt.append("alpha_s",value=0.2,min=0,max=2,round=2)
    opt.config()
    #opt.minimize({'target':"y3",'iteration':10,'population':10,'samples':10})
    opt.minimize({'TARGET':func,'iteration':10,'population':20,'samples':20})
    exe = opt.exe_best
#optimize()

### グリッドサーチ

st.exe = exe

def exe1(id):
    csv = prefix+id+"_exe1.csv"
    png = prefix+id+"_exe1.png"
    st.execute1(exe,file_csv=csv,file_fig1=png)

def gridsearch(X1,min=0,max=1,num=41,samples=10):
    csv = prefix+"_scan1ds_"+X1+".csv"
    png = prefix+"_scan1ds_"+X1+".png"

    #print("scan1ds...")
    st.scan1ds(csv,X1,min=min,max=max,num=num,samples=samples,run=1,list=0)

    #print("loading...")
    df = pd.read_csv(csv,sep=',',names=columns)

    #print("plotting...")
    plt.figure()

    plt.subplot(2,1,1)
    x,ymean,ystd,ymin,ymax = st.analyze(df,X1,"RMSE1")
    plt.errorbar(x,ymean,yerr=ystd,fmt='o',color='b',ecolor='gray',capsize=2)
    plt.ylabel('RMSE1')

    plt.subplot(2,1,2)
    #x,ymean,ystd,ymin,ymax = st.analyze(df,X1,"count_gap")
    #plt.errorbar(x,ymean,yerr=ystd,fmt='o',color='b',ecolor='gray',capsize=2)
    #plt.ylabel('gap')

    x,ymean,ystd,ymin,ymax = st.analyze(df,X1,"overflow")
    plt.errorbar(x,ymean,yerr=ystd,fmt='o',color='b',ecolor='gray',capsize=2)
    plt.ylabel('overflow')

    plt.xlabel(X1)

    st.plt_output(png)

#exe1("a")
#optimize()
#gridsearch("alpha_r",min=0,max=1,num=51,samples=10)
#gridsearch("alpha_i",min=0,max=1,num=51,samples=10)
gridsearch("alpha_s",min=0,max=2,num=51,samples=10)
#gridsearch("Temp",min=0.01,max=2,num=51,samples=10)
#gridsearch("beta_i",min=0,max=1,num=51,samples=10)
#gridsearch("beta_r",min=0,max=1,num=51,samples=10)
