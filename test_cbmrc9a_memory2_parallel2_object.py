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
from cbmrc9a_memory2_parallel2_object import Config
config = Config()
common.config  = config
common.prefix  = "data%s_cbmrc9a_memory2_parallel2_object" % common.string_now() # 実験名（ファイルの接頭辞）
common.dir_path= "data/data%s_cbmrc9a_memory2_parallel2_object" % common.string_now() # 実験データを出力するディレクトリのパス
common.exe     = "python cbmrc9a_memory2_parallel2_object.py " # 実行されるプログラム
common.columns=['dataset','seed','id','NN','Nh',\
    'alpha_i','alpha_r','alpha_b','alpha_s',\
    # 'alpha_i2','alpha_r2','alpha_s2',\
    # 'alpha_i3','alpha_r3','alpha_s3',\
    # "alpha1",\
    # 'beta_i','beta_r',\
    # 'beta_i2','beta_r2',\
    # 'beta_i3','beta_r3',
    'beta_b',\
        'Temp','lambda0',"delay","parallel",\
        'RMSE1','RMSE2','cnt_overflow','MC']
common.parallel= 8
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
    opt.append("beta_i",value=0.01,min=0.0,max=1,round=2)
    opt.append("alpha_i",value=1,min=0.00,max=1,round=2)
    opt.append("alpha_r",value=1,min=0.,max=1,round=2)
    opt.append("alpha_s",value=1,min=0,max=2,round=2)

    #opt.append("alpha1",value=1,min=0,max=2,round=2)
    #opt.append("Temp",value=10,min=1,max=10,round=2)
    opt.maximize(target="MC",iteration=30,population=32,samples=2)
    common.config = opt.best_config # 最適化で得られた設定を基本設定とする
#optimize()

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
    # gridsearch("alpha_r",min=0.00,max=1,num=41,samples=ns)
    # gridsearch("alpha_i",min=0.00,max=1,num=41,samples=ns)
    # gridsearch("alpha_s",min=0.0,max=2,num=41,samples=ns)
    # gridsearch("beta_r",min=0.00,max=1,num=41,samples=ns)
    # gridsearch("beta_i",min=0.00,max=1,num=41,samples=ns)
    #gridsearch("Temp",min=-256,max=256,num=4001,samples=ns)
    #gridsearch("alpha1",min=0.0,max=2,num=41,samples=ns)
    #gridsearch("delay",min=5,max=100,num=41,samples=ns)
    #gridsearch("lambda0",min=0.01,max=1.5,num=41,samples=ns)
#gs2()

def gs2():
    x1,x2 = "Nh","parallel"
    gs.scan2d(x1,x2,min1=20,max1=100,min2=1,max2=10,samples=3,num1=11,num2=11)
    vs.plot2d(x1,x2,"MC")
    vs.plot2ds_pcolor(x1,x2,"MC")
#gs2()

# csv = '/home/yamato/Downloads/cbm_rc/data/data20220211_183544_cbmrc9a_memory2_parallel2_object/data20220211_183544_cbmrc9a_memory2_parallel2_object_scan2d_Nh_parallel.csv'

# x1,x2 = "Nh","parallel"
# # vs.plot2ds_pcolor(x1,x2,"MC",csv=csv)
# fig = None

# if fig==None:
#     fig=os.path.splitext(common.last_csv)[0] + '_pcolor.png'
# df = common.load_dataframe(csv)
# df = df.sort_values('id',ascending=True)
# x1,x2,ymean,ystd,ymin,ymax = vs.analyze2d(df,"id",x1,x2,"MC")
# nx1=len(set(x1))
# nx2=len(set(x2))+1
# x1=x1.reshape(nx1,nx2)
# x2=x2.reshape(nx1,nx2)
# ymean=ymean.reshape(nx1,nx2)

# plt.figure()
# plt.pcolor(x1,x2,ymean)
# plt.colorbar()
# plt.xlabel("Nh")
# plt.ylabel("parallel")
# plt.title("MC")
# #plt.show()
# vs.plt_output(fig)