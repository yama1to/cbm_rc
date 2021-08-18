# Copyright (c) 2018 Katori lab. All Rights Reserved
# NOTE:
import os
import datetime
import numpy as np
import pandas as pd
import matplotlib as mpl
#mpl.use('Agg') ## サーバ上で画像を出力するための設定。ローカルで表示する際はコメントアウト
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import optimization as opt
import searchtools as st

### 共通設定
string_now = datetime.datetime.now().strftime("%Y%m%d") # 年月日
exe = "python model5.py display=0 " # 基本コマンド、ここにパラメータの設定してプログラムが実行される。
prefix = "data%s_test1_model5" % (string_now) # 実験名(data[日付]_[実験番号]_[モデル名])
path = prefix # 実験データを出力するディレクトリのパス
report = prefix + ".md" # マークダウンファイルを名前
columns=['id','seed','x1','x2','x3','y1','y2','y3']

### searchtools(ランダムサーチ・グリッドサーチ)の基本設定
st.set_path(path)
st.exe = exe
st.columns = columns
st.file_md = report
st.parallel = 8
st.report("## model5 %s\n" % (prefix))

### 最適化
def func(row):
    return row['y1'] + 0.3*row['y2']

def optimize():
    global exe
    opt.set_path(path)
    opt.exe = exe
    opt.columns = columns
    opt.parallel = 8
    opt.file_md = report
    opt.file_opt = prefix + "_opt.csv"
    opt.clear()#設定をクリアする
    opt.appendid()#id:必ず加える
    opt.appendseed()# 乱数のシード（０から始まる整数値）
    opt.append("x1",value=1.0,min=-5,max=5,round=3)#([変数名],[基本値],[下端],[上端],[まるめの桁数])
    opt.append("x2",value=1.0,min=-5,max=5)
    opt.minimize({'target':"y1",'iteration':2,'population':10,'samples':2})
    #opt.minimize({'TARGET':func,'iteration':10,'population':10,'samples':4}) # 関数funcでtargetを指定する。
    exe = opt.exe_best #　最適化で得られたパラメータ値を基本コマンドに追加する。

### 単体実行
#exe += "x1=1.23 x2=-4.56 "
def exe1(id=""):
    csv = prefix+id+"_exe1.csv"
    png = prefix+id+"_exe1.png"
    st.execute1(exe,file_csv=csv,file_fig1=png)


### ランダムサーチ
def rs():
    csv = prefix+"_random.csv"
    png = prefix+"_random.png"
    st.clear()
    st.append("x1",min=-5,max=5)
    st.append("x2",min=-5,max=5)
    st.random(csv,num=100,run=1,list=0) #ランダムサーチを実行し、結果をCSVに保存する
    df = st.load_dataframe(csv) # csvファイルを読み込む
    df = df[['x1','x2','y1','y2']] # 指定した列のみでデータフレームを構成する
    df = df[(df['y1']<=10.0)] # 条件を満たすデータについてデータフレームを構成する。
    #print(df)
    scatter_matrix(df, alpha=0.8, figsize=(6, 6), diagonal='kde')
    st.plt_output(png)

### １次元グリッドサーチ
def gs1d(X1,min=0,max=1,num=41,samples=10):
    csv = prefix+"_scan1ds_"+X1+".csv"
    png = prefix+"_scan1ds_"+X1+".png"
    st.exe = exe
    st.scan1ds(csv,X1,min=min,max=max,num=num,samples=samples,run=1,list=0)

    df = st.load_dataframe(csv)

    cmap = plt.get_cmap("tab10")
    plt.figure()
    x,ymean,ystd,ymin,ymax = st.analyze(df,X1,"y1")# "y1"の平均、標準偏差、最小値、最大値計算
    #plt.errorbar(x,ymean,yerr=ystd,fmt='o',color=cmap(0),capsize=2,label="e0")
    plt.plot(x,ymean,color=cmap(0),marker='o',linewidth=0,label="y1")
    plt.plot(x,ymin,color=cmap(0),linewidth=1)
    plt.plot(x,ymax,color=cmap(0),linewidth=1)

    x,ymean,ystd,ymin,ymax = st.analyze(df,X1,"y2")
    #plt.errorbar(x,ymean,yerr=ystd,fmt='o',color=cmap(1),capsize=2,label="e1")
    plt.plot(x,ymean,color=cmap(1),marker='o',linewidth=0,label="y2")
    plt.plot(x,ymin,color=cmap(1),linewidth=1)
    plt.plot(x,ymax,color=cmap(1),linewidth=1)

    plt.ylabel("y1,y2")
    plt.xlabel(X1)
    plt.legend()
    #plt.ylim([0,1])
    st.plt_output(png)

def gs1():
    gs1d("x1",min=-5,max=5,num=41,samples=10)
    gs1d("x2",min=-5,max=5,num=41,samples=10)

### 2次元グリッドサーチ
def gs2():
    st.exe = exe
    csv = prefix+"_scan2ds_x1_x2.csv"
    png1 = prefix+"_scan2ds_x1_x2_plot.png"
    png2 = prefix+"_scan2ds_x1_x2_pcolor.png"
    st.scan2ds(csv,"x1","x2",min1=-5,max1=5,num1=11,min2=-5,max2=5,num2=11,samples=3,run=1,list=0)
    st.plot2ds(csv,"x1","x2","y3",fig=png1)
    st.plot2ds_pcolor(csv,"x1","x2","y3",fig=png2)

exe1("a")
optimize()
exe1("b")
rs()
gs1()
gs2()
