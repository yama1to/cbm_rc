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

#from esn_memory4 import Config
#global Config
def esn_optimize(Config,iteration,population,samples):
    config = Config
    
    common.config  = config
    common.prefix  = "data%s_esn_memory4" % common.string_now() # 実験名（ファイルの接頭辞
    common.dir_path= "data/data%s_esn/data%s_esn_memory4" % (common.string_today(),common.string_now()) # 実験データを出力するディレクトリのパス
    common.exe     = "python esn_memory4.py " # 実行されるプログラム
    common.columns =['dataset','seed','id','Nh','alpha_i','alpha_r','alpha0',
    'beta_i','beta_r',"delay",'lambda0',"MC"]
    common.parallel= 2
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
        opt.append("beta_r",value=0.1,min=0.01,max=1,round=2)
        opt.append("beta_i",value=0.1,min=0.01,max=1,round=2)
        opt.append("alpha_i",value=1,min=0.1,max=1,round=2)
        opt.append("alpha_r",value=0.9,min=0.7,max=1,round=2)
        opt.minimize(target="MC",iteration=iteration,population=population,samples=samples)
        #opt.minimize(TARGET=func,iteration=5,population=10,samples=4)

        common.config = opt.best_config # 最適化で得られた設定を基本設定とする

    optimize(iteration,population,samples)
    return common.config.MC

