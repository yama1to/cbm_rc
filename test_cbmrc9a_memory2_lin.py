# Copyright (c) 2018-2021 Katori lab. All Rights Reserved
"""

"""
import csv
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
from cbmrc9a_memory2_lin import Config
config = Config()
common.config  = config
common.prefix  = "data%s_cbmrc9a_memory2_lin" % common.string_now() # 実験名（ファイルの接頭辞）
common.dir_path= "data/data%s_cbmrc9a_memory2_lin" % common.string_now() # 実験データを出力するディレクトリのパス
common.exe     = "python cbmrc9a_memory2_lin.py " # 実行されるプログラム
common.columns=['dataset','seed','id','NN','Nh','alpha_i','alpha_r','alpha_r2','alpha_b','alpha_s',"alpha0",'beta_i','beta_r','beta_r2','beta_b',
'Temp','lambda0',"delay",'RMSE1','RMSE2','cnt_overflow','MC']
common.parallel= 90
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
    df = df[['alpha_r','alpha_i','cnt_overflow','MC']] # 指定した列のみでデータフレームを構成する
    #df = df[(df['y1']<=10.0)] # 条件を満たすデータについてデータフレームを構成する。
    #print(df)
    scatter_matrix(df, alpha=0.8, figsize=(6, 6), diagonal='kde')
    vs.savefig()
#rs1()

### 最適化
def func(row):# 関数funcでtargetを指定する。
    return row['y1'] + 0.3*row['y2']

def optimize():
    opt.clear()#設定をクリアする
    opt.appendid()#id:必ず加える
    opt.appendseed()# 乱数のシード（０から始まる整数値）
    opt.append("beta_r",value=0.01,min=0.0,max=1,round=2)
    opt.append("beta_r2",value=0.01,min=0.0,max=1,round=2)
    opt.append("beta_i",value=0.01,min=0.0,max=1,round=2)
    opt.append("alpha_i",value=1,min=0.00,max=1,round=2)
    opt.append("alpha_r",value=1,min=0.,max=1,round=2)
    opt.append("alpha_r2",value=1,min=0.,max=1,round=2)
    opt.append("alpha_s",value=1,min=0,max=2,round=2)
    #opt.append("alpha0",value=1,min=0,max=1,round=2)
    #opt.append("alpha0",value=alpha0,min=alpha0,max=alpha0,round=2)
    #opt.append("Temp",value=10,min=1,max=10,round=2)
    opt.maximize(target="MC",iteration=30,population=30,samples=3)
    common.config = opt.best_config # 最適化で得られた設定を基本設定とする

optimize()

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

def gs1():
    ns=3
    #gridsearch("Nh",min=50,max=700,num=41,samples=ns)
    gridsearch("alpha0",min=0.0,max=1,num=41,samples=ns)
    gridsearch("alpha_r",min=0.0,max=1,num=41,samples=ns)
    gridsearch("alpha_r2",min=0.0,max=1,num=41,samples=ns)
    gridsearch("alpha_i",min=0.0,max=1,num=41,samples=ns)
    gridsearch("alpha_s",min=0.0,max=2,num=41,samples=ns)
    
    gridsearch("beta_r",min=0.0,max=1,num=41,samples=ns)
    gridsearch("beta_r2",min=0.0,max=1,num=41,samples=ns)
    gridsearch("beta_i",min=0.0,max=1,num=41,samples=ns)
    #gridsearch("delay",min=5,max=100,num=41,samples=ns)
    #gridsearch("lambda0",min=0.01,max=1.5,num=41,samples=ns)
gs1()
def gs2():
    x1,x2 = "alpha_r2","alpha0"
    gs.scan2d(x1,x2,min1=0,max1=1,min2=0,max2=1,samples=3,num1=21,num2=21)
    vs.plot2d(x1,x2,"MC")
    vs.plot2ds_pcolor(x1,x2,"MC")
# gs2()
# file = '/home/yamato/Downloads/cbm_rc/data/data20220206_093733_cbmrc9a_memory2_lin/data20220206_093733_cbmrc9a_memory2_lin_scan2d_alpha_r_alpha0.csv'
# vs.plot2d("alpha_r","alpha0","MC",fig="2ds1",csv =file)
# vs.plot2ds_pcolor("alpha_r","alpha0","MC",fig="2ds2",csv=file)

# global alpha0
# for alpha0 in range(10):
#     alpha0 /=10
#     optimize()
#     gs2()
