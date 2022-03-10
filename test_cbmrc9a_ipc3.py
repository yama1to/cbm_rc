# Copyright (c) 2018-2021 Katori lab. All Rights Reserved
"""

"""
import os
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

from explorer import common
from explorer import gridsearch as gs
from explorer import visualization as vs
from explorer import randomsearch as rs
from explorer import optimization as opt

### 共通設定
from cbmrc9a_ipc3 import Config
config = Config()
common.config  = config
common.prefix  = "data%s_cbmrc9a_ipc3" % common.string_now() # 実験名（ファイルの接頭辞）
common.dir_path= "data/data%s_cbmrc9a_ipc3" % common.string_now() # 実験データを出力するディレクトリのパス
common.exe     = "python cbmrc9a_ipc3.py " # 実行されるプログラム
common.columns=['dataset','seed','id','NN','Nh','alpha_i','alpha_r','alpha_b','alpha_s','beta_i','beta_r','beta_b',
'Temp','lambda0',"delay","degree","MC","set",'cnt_overflow',"per"]
common.parallel= 32
common.setup()
common.report_common()
common.report_config(config)

### ランダムサーチ
def rs1():
    rs.clear()
    rs.append("alpha_r",min=0,max=5)
    rs.append("alpha_i",min=0,max=5)
    rs.random(num=60,samples=2)
    df = common.load_dataframe() # 直前に保存されたcsvファイルをデータフレーム(df)に読み込む
    df = df[['alpha_r','alpha_i','cnt_overflow',"MC"]] # 指定した列のみでデータフレームを構成する
    #df = df[(df['y1']<=10.0)] # 条件を満たすデータについてデータフレームを構成する。
    #print(df)
    scatter_matrix(df, alpha=0.8, figsize=(6, 6), diagonal='kde')
    vs.savefig()
#rs1()

### 最適化
def func(row):# 関数funcでtargetを指定する。
    return row['y1'] + 0.3*row['y2']

def optimize(i,x):
    opt.clear()#設定をクリアする
    opt.appendid()#id:必ず加える
    opt.appendseed()# 乱数のシード（０から始まる整数値）
    #opt.append("Nh",value=500,min=300,max=1000,round=1)
    opt.append("beta_r",value=0.01,min=0.01,max=1,round=2)
    opt.append("beta_i",value=0.01,min=0.01,max=1,round=2)
    opt.append("alpha_i",value=1,min=0.01,max=1,round=2)
    opt.append("alpha_r",value=1,min=0.,max=1,round=2)
    opt.append("alpha_s",value=1,min=0,max=2,round=2)
    #opt.append("degree",value=i,min=i,max=i,round=0)
   #opt.append("set",value=x,min=x,max=x,round=0)
    #opt.append("Temp",value=10,min=1,max=10,round=2)
    opt.maximize(target="MC",iteration=20,population=20,samples=3)
    #opt.minimize(TARGET=func,iteration=5,population=10,samples=4)
    common.config = opt.best_config # 最適化で得られた設定を基本設定とする
for x in range(1):
    for i in range(10):
        optimize(i+1,x)




def plot1(x,y,ystd,ymin,ymax,color=None,width=1,label=None):
    # エラーバーをつけてグラフを描画、平均、標準偏差、最大値、最小値をプロットする。
    #ax.errorbar(x,y,yerr=ystd,fmt='o',color=color,capsize=2,label="xxxx")
    plt.plot(x,y,color=color,linestyle='-',linewidth=width,label=label)
    plt.fill_between(x,y-ystd,y+ystd,color=color,alpha=.2)
    plt.plot(x,ymin,color=color,linestyle=':',linewidth=1)
    plt.plot(x,ymax,color=color,linestyle=':',linewidth=1)


def gridsearch(X1,min=0,max=1,num=41,samples=10):
    # 指定された変数(X1)についてグリッドサーチを行い、評価基準の変化をまとめてプロット

    gs.scan1ds(X1,min=min,max=max,num=num,samples=samples)
    df = common.load_dataframe()
    #print(df)
    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(6,8))

    plt.subplot(2,1,1)
    x,ymean,ystd,ymin,ymax = vs.analyze(df,X1,"MC")
    plot1(x,ymean,ystd,ymin,ymax,color=cmap(1),label="MC")
    plt.ylabel("MC")
    plt.grid(linestyle="dotted")

    plt.subplot(2,1,2)
    x,ymean,ystd,ymin,ymax = vs.analyze(df,X1,"cnt_overflow")
    plot1(x,ymean,ystd,ymin,ymax,color=cmap(2),label="cnt_overflow")
    plt.ylabel("overflow")
    #plt.yscale('log')
    plt.grid(linestyle="dotted")
    #plt.ylim([0,1]) # y軸の範囲

    plt.xlabel(X1)
    vs.plt_output()
    



def gs2():
    ns=10
    #gridsearch("Nh",min=50,max=300,num=41,samples=ns)
    #gridsearch("delay",min=1,max=20,num=10,samples=ns)
    gridsearch("degree",min=1,max=10,num=11,samples=ns)
    #gridsearch("alpha_i",min=0.1,max=1,num=10,samples=ns)
    # gridsearch("alpha_r",min=0,max=1,num=41,samples=ns)
    # gridsearch("alpha_s",min=0.01,max=1,num=41,samples=ns)
    # gridsearch("beta_i",min=0.01,max=1,num=41,samples=ns)
    # gridsearch("beta_r",min=0.01,max=1,num=41,samples=ns)
    # gridsearch("Temp",min=1,max=10,num=41,samples=ns)
    #plt.legend()
    #plt.show()
#gs2()
