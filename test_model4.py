# Copyright (c) 2018-2019 Katori lab. All Rights Reserved
"""
explorerのテストコード（変数に文字列を含む場合）

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
from main_model4 import Config
config = Config()
common.config  = config
common.prefix  = "data%s_model4" % common.string_today() # 実験名（ファイルの接頭辞）
common.dir_path= "data/data%sa_model4" % common.string_today() # 実験データを出力するディレクトリのパス
common.exe     = "python main_model4.py " # 実行されるプログラム
common.columns = ['id','seed','x1','x2','x3','f1','y1','y2','y3']
common.parallel= 4
common.setup()
common.report_common()
common.report_config(config)

### 単体実行
def exe1():# 基本設定で実行
    gs.execute()
#exe1()

def exe2():# 基本設定を編集してから実行
    cnf=copy.copy(config) #設定(config)のコピーを作成する
    cnf.x1 = 1.234 # 設定を書き換える
    cnf.f1 = 'tanh'
    gs.execute(config=cnf) # 新しい設定(cnf)を反映させて実行する。
exe2()

def gs1d():# 文字列リストによるグリッドサーチ
    a = ['rel','tanh','sin']
    gs.scan1d("f1",array=a)
    vs.plot1d("f1","y3")
gs1d()

def gs2a():
    gs.scan2d("x1","x2",min1=-5,max1=5,min2=-5,max2=5)
    vs.plot2d("x1","x2","y3")
    #vs.plot2d_pcolor("x1","x2","y3")
gs2a()

def gs2b():
    gs.scan2d("x1","x2",min1=-5,max1=5,min2=-5,max2=5,samples=3)
    vs.plot2d("x1","x2","y3")
gs2b()

# TODO 文字列・整数値を含むランダムサーチとランダムサーチ

### ランダムサーチ
def rs1():
    rs.clear()
    rs.append("x1",min=-5,max=5)
    rs.append("x2",min=-5,max=5)
    rs.random(num=2000,samples=2)
    df = common.load_dataframe() # 直前に保存されたcsvファイルをデータフレーム(df)に読み込む
    df = df[['x1','x2','y1','y2']] # 指定した列のみでデータフレームを構成する
    df = df[(df['y1']<=10.0)] # 条件を満たすデータについてデータフレームを構成する。
    #print(df)
    scatter_matrix(df, alpha=0.8, figsize=(6, 6), diagonal='kde')
    fig=common.name_file(common.prefix+"_random.png")
    vs.savefig(fig)
#rs1()

### 最適化
def func(row):# 関数funcでtargetを指定する。
    return row['y1'] + 0.3*row['y2']

def optimize():
    opt.clear()#設定をクリアする
    opt.appendid()#id:必ず加える
    opt.appendseed()# 乱数のシード（０から始まる整数値）
    opt.append("x1",value=1.0,min=-5,max=5,round=3)# 変数の追加([変数名],[基本値],[下端],[上端],[まるめの桁数])
    opt.append("x2",value=1.0,min=-5,max=5)
    opt.minimize(target="y1",iteration=10,population=10,samples=4)
    #opt.minimize(TARGET=func,iteration=5,population=10,samples=4)
#optimize()
