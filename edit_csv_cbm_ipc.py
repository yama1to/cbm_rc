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
    x,ymean,ystd,ymin,ymax = vs.analyze(df,X1,"WER")
    plot1(x,ymean,ystd,ymin,ymax,color=cmap(1),label="WER")
    plt.ylabel("WER")
    plt.grid(linestyle="dotted")

    plt.ylim([-0,0.3]) # y軸の範囲

    
    plt.xlabel(X1)
    vs.plt_output()

if __name__ == "__main__":
    #global csv 
    ### 共通設定
    from cbmrc9a_ipc3 import Config
    config = Config()
    common.config  = config
    common.prefix  = "data%s_cbmrc9a_ipc3" % common.string_now() # 実験名（ファイルの接頭辞）
    common.dir_path= "data/data%s_cbmrc9a_ipc3" % common.string_now() # 実験データを出力するディレクトリのパス
    common.exe     = "python cbmrc9a_ipc3.py " # 実行されるプログラム
    common.columns=['dataset','seed','id','NN','Nh','alpha_i','alpha_r','alpha_b','alpha_s','beta_i','beta_r','beta_b',
    'Temp','lambda0',"delay","degree","set",'cnt_overflow','CAPACITY',"per"]
    'lambda0',"train_WER","WER"]
    common.parallel= 32
    common.setup()
    common.report_common()
    common.report_config(config)

    d = "data20211208_160838_cbmrc9a_ipc3_scan1d_alpha_i.csv"
    dir = '/home/yamato/Downloads/cbm_rc/data/data20211208_160838_cbmrc9a_ipc3/'


    labels = ["alpha_i","alpha_r","beta_i","beta_r",]
    
    for name,label in zip(d,labels):
        gridsearch(name,dir,label)
        print(name,label)
    # for name,label in zip(csv_name,labels):
    #     csv = dir+"/"+name
    #     gridsearch(label,csv)

    