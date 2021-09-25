# Copyright (c) 2018-2021 Katori lab. All Rights Reserved
"""

"""
import os
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import csv

from explorer import common
from explorer import gridsearch as gs
from explorer import visualization as vs
from explorer import randomsearch as rs
from explorer import optimization as opt

#from esn_approximation2 import Config
#global Config
def esn_optimize(Config,iteration,population,samples):
    config = Config
    
    common.config  = config
    common.prefix  = "data%s_esn_approximation2" % common.string_now() # 実験名（ファイルの接頭辞
    common.dir_path= "data/data%s_esn/data%s_esn_approximation2" % (common.string_today(),common.string_now()) # 実験データを出力するディレクトリのパス
    common.exe     = "python esn_approximation2.py " # 実行されるプログラム
    common.columns =['dataset','seed','id','Nh','alpha_i','alpha_r','alpha0',
    'beta_i','beta_r',"delay","logv","f",'lambda0',"RMSE1","NRMSE"]
    common.parallel= 1
    #common.dir_path = "/Users/yamato/pypr/cbm_rc/"
    common.setup()
    common.report_common()
    common.report_config(config)

    
    ### 最適化
    def func(row):# 関数funcでtargetを指定する。
        return row['y1'] + 0.3*row['y2']

    def optimize(iteration,population,samples):
        opt.clear()#設定をクリアする
        opt.appendid()#id:必ず加える
        opt.appendseed()# 乱数のシード（０から始まる整数値
        
        #opt.append("alpha0",value=1,min=0.01,max=1,round=2)
        opt.append("beta_r",value=0.01,min=0.01,max=1,round=2)
        opt.append("beta_i",value=0.01,min=0.01,max=1,round=2)
        opt.append("alpha_i",value=1,min=0.01,max=1,round=2)
        opt.append("alpha_r",value=0.9,min=0.7,max=1,round=2)
        opt.append("alpha0",value=1,min=0.01,max=1,round=2)
        opt.minimize(target="NRMSE",iteration=iteration,population=population,samples=samples)
        #opt.minimize(TARGET=func,iteration=5,population=10,samples=4)

        common.config = opt.best_config # 最適化で得られた設定を基本設定とする

    optimize(iteration,population,samples)
    return common.config.NRMSE

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
        x,ymean,ystd,ymin,ymax = vs.analyze(df,X1,"NRMSE")
        plot1(x,ymean,ystd,ymin,ymax,color=cmap(1),label="NRMSE")
        plt.ylabel("NRMSE")
        plt.grid(linestyle="dotted")

        plt.subplot(2,1,2)
        x,ymean,ystd,ymin,ymax = vs.analyze(df,X1,"RMSE1")
        plot1(x,ymean,ystd,ymin,ymax,color=cmap(2),label="RMSE1")
        plt.ylabel("RMSE")
        #plt.yscale('log')
        plt.grid(linestyle="dotted")
        #plt.ylim([0,1]) # y軸の範囲

        plt.xlabel(X1)
        vs.plt_output()

    def gs2():
        ns=3
        #gridsearch("Nh",min=100,max=1000,num=41,samples=ns)
        gridsearch("alpha_r",min=0.01,max=2,num=41,samples=ns)
        gridsearch("alpha_i",min=0.01,max=2,num=41,samples=ns)
    gs2()
