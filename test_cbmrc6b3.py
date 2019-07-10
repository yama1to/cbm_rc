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
columns=['dataset','seed','id','NN','Nh','alpha_i','alpha_r','alpha_b','alpha_s','alpha0','alpha1','beta_i','beta_r','beta_b','Temp','lambda0','RMSE1','RMSE2','count_gap']
exe = "python cbmrc6b3.py display=0 dataset=%d " % (dataset)
report = "data20181220_cbmrc6b3.md"
prefix = "data20181220_cbmrc6b3_dataset%d" % (dataset)

st.exe = exe
st.columns = columns
st.file_md = report
text="## cbmrc6b3 (%s)  \n"  % (prefix)
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
    opt.append("alpha_r",value=0.1,min=0,max=1,round=2)
    opt.append("alpha_i",value=0.1,min=0,max=1,round=2)
    opt.append("alpha_s",value=0.2,min=0,max=2,round=2)
    opt.config()
    #opt.minimize({'target':"y3",'iteration':10,'population':10,'samples':10})
    opt.minimize({'TARGET':func,'iteration':10,'population':20,'samples':20})
optimize()

### グリッドサーチ

exe+= "alpha_r=%f alpha_i=%f alpha_s=%f " % (opt.xbest['alpha_r'],opt.xbest['alpha_i'],opt.xbest['alpha_s'])
st.exe = exe

png = prefix+"_test.png"
st.execute1(exe,file_fig1=png)

def gridsearch(X1,min=0,max=1,num=41,samples=10):
    '''
    cbmrc6 についてグリッドサーチを行う
    '''

    csv = prefix+"_scan1ds_"+X1+".csv"
    png = prefix+"_scan1ds_"+X1+".png"
    st.scan1ds(csv,X1,min=min,max=max,num=num,samples=samples,run=1,list=0)

    df = pd.read_csv(csv,sep=',',names=columns)

    plt.figure()

    plt.subplot(2,1,1)
    x,ymean,ystd,ymin,ymax = st.analyze(df,X1,"RMSE1")
    plt.errorbar(x,ymean,yerr=ystd,fmt='o',color='b',ecolor='gray',capsize=2)
    plt.ylabel('RMSE1')

    plt.subplot(2,1,2)
    x,ymean,ystd,ymin,ymax = st.analyze(df,X1,"count_gap")
    plt.errorbar(x,ymean,yerr=ystd,fmt='o',color='b',ecolor='gray',capsize=2)
    plt.ylabel('gap')

    plt.xlabel(X1)

    st.plt_output(png)
'''
gridsearch("alpha_r",min=0,max=1,num=51,samples=20)
gridsearch("alpha_i",min=0,max=1,num=51,samples=20)
gridsearch("alpha_s",min=0,max=1,num=51,samples=20)
gridsearch("beta_i",min=0,max=1,num=51,samples=20)
gridsearch("beta_r",min=0,max=1,num=51,samples=20)
gridsearch("Temp",min=0.01,max=2,num=51,samples=20)
'''
