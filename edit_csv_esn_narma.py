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

def plot1(x,y,ystd,ymin,ymax,color=None,width=1,label=None):
    # エラーバーをつけてグラフを描画、平均、標準偏差、最大値、最小値をプロットする。
    #ax.errorbar(x,y,yerr=ystd,fmt='o',color=color,capsize=2,label="xxxx")
    plt.plot(x,y,color=color,linestyle='-',linewidth=width,label=label)
    plt.fill_between(x,y-ystd,y+ystd,color=color,alpha=.2)
    plt.plot(x,ymin,color=color,linestyle=':',linewidth=1)
    plt.plot(x,ymax,color=color,linestyle=':',linewidth=1)

def gridsearch(csv,path,label):
    # 指定された変数(X1)についてグリッドサーチを行い、評価基準の変化をまとめてプロット
    X1 = label
    #gs.scan1ds(X1,min=min,max=max,num=num,samples=samples)

    df = common.load_dataframe(csv = csv,path = path)
    #print(df)
    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(6,4))

    plt.subplot(1,1,1)
    x,ymean,ystd,ymin,ymax = vs.analyze(df,X1,"NMSE")
    plot1(x,ymean,ystd,ymin,ymax,color=cmap(1),label="NMSE")
    plt.ylabel("NMSE")
    plt.grid(linestyle="dotted")
    plt.xlim([0,1])
    plt.ylim([-0.1,2]) # y軸の範囲

    
    plt.xlabel(X1)
    vs.plt_output()

if __name__ == "__main__":
    #global csv 
    from esn_narma import Config
    config = Config()
    common.config  = config
    common.prefix  = "data%s_esn_narma" % common.string_now() # 実験名（ファイルの接頭辞）
    common.dir_path= "data/data%s_esn_narma" % common.string_now() # 実験データを出力するディレクトリのパス
    common.exe     = "python esn_narma.py " # 実行されるプログラム
    common.columns =['dataset','seed','id','Nh','alpha_i','alpha_r','alpha_b','beta_i','beta_r','beta_b','lambda0',"RMSE",'NRMSE',"NMSE"]
    common.parallel= 100
    common.setup()
    common.report_common()
    common.report_config(config)

    dir = "/home/yamato/Desktop/optimizeList/out3/data20211015_182129_esn_narma/"
    csv_name = ["data20211015_182129_esn_narma_scan1d_alpha_i.csv",
                "data20211015_182129_esn_narma_scan1d_alpha_r.csv",
                "data20211015_182129_esn_narma_scan1d_beta_i.csv",
                "data20211015_182129_esn_narma_scan1d_beta_r.csv"
                ]
    labels = ["alpha_i","alpha_r","beta_i","beta_r"]
    
    for name,label in zip(csv_name,labels):
        gridsearch(name,dir,label)
        print(name,label)
    # for name,label in zip(csv_name,labels):
    #     csv = dir+"/"+name
    #     gridsearch(label,csv)

    