# Copyright (c) 2018 Katori lab. All Rights Reserved

import subprocess
import sys
import numpy as np
import pandas as pd
import itertools
from pprint import pprint
import matplotlib as plt
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

def execute(commands):# コマンドのリストを受け取り、parallel で実行する
    filename_sh="cmd.sh"
    f=open(filename_sh,"w")
    for command in commands:
        f.write(command+"\n")
    f.close()
    command="parallel -a %s" % filename_sh
    subprocess.call(command.split())

def scan_1d(exe,file,X1,min,max,num):#１変数のスキャン
    c=[]
    id=0
    for x1 in np.linspace(min,max,num):
        c.append(exe+" file=%s id=%d %s=%f " % (file,id,X1,x1))
        id+=1
    return c

def scan_s1d(exe,file,num_seed,X1,min,max,num):# seedと１変数のスキャン
    c=[]
    for seed in np.arange(num_seed):
        id=0
        for x1 in np.linspace(min,max,num):
            c.append(exe+" file=%s seed=%d id=%d %s=%f " % (file,seed,id,X1,x1))
            id+=1
    return c

def scan_2d(exe,file,X1,min1,max1,num1,X2,min2,max2,num2):#２変数によるスキャン
    c=[]
    id=0
    for x1 in np.linspace(min1,max1,num1):
        for x2 in np.linspace(min2,max2,num2):
            c.append(exe+" file=%s id=%d %s=%f %s=%f" % (file,id,X1,x1,X2,x2))
            id+=1
    return c

def scan_s2d(exe,file,num_seed,X1,min1,max1,num1,X2,min2,max2,num2):#　seedと２変数によるスキャン
    c=[]
    for seed in np.arange(num_seed):
        id=0
        for x1 in np.linspace(min1,max1,num1):
            for x2 in np.linspace(min2,max2,num2):
                c.append(exe+"file=%s seed=%d id=%d %s=%f %s=%f" % (file,seed,id,X1,x1,X2,x2))
                id+=1
    return c

def analyze(df0,key,y):
    #df0= pd.read_csv(file,sep=',',names=columns)
    df = df0.groupby(key).aggregate(['mean','std','min','max'])
    x1 = df0.groupby(key)[key].mean().values
    y1mean = df[y,'mean'].values
    y1std  = df[y,'std'].values
    y1min  = df[y,'min'].values
    y1max  = df[y,'max'].values
    return x1,y1mean,y1std,y1min,y1max

def analyze2d(df0,key,X1,X2,Y):
    #df0= pd.read_csv(file,sep=',',names=columns)
    df = df0.groupby(key).aggregate(['mean','std','min','max'])
    #x1 = df0.groupby(key)[x1].mean().values
    #x2 = df0.groupby(key)[x2].mean().values
    y1mean = df[Y,'mean'].values
    y1std  = df[Y,'std'].values
    y1min  = df[Y,'min'].values
    y1max  = df[Y,'max'].values
    x1 = df[X1,'mean'].values
    x2 = df[X2,'mean'].values
    return x1,x2,y1mean,y1std,y1min,y1max

def plot_1d(file,columns,X1,Y1):
    """
    1次元プロットを作成する。
    file: データが含まれるCSVファイル
    columns: CSVファイルの列（コラム)の名前の一覧
    X1: x軸の列名(columnsの中に含まれる名前)
    Y1: y軸の列名(columnsの中に含まれる名前)
    """
    df = pd.read_csv(file,sep=',',names=columns)
    plt.plot(df[X1],df[Y1],'o')
    plt.ylabel(Y1)
    plt.xlabel(X1)
    plt.show()

def plot_s1d(file,columns,X1,Y1):
    """
    1次元プロットを作成する。異なるseedのデータについては平均を標準偏差をとる
    file: データが含まれるCSVファイル
    columns: CSVファイルの列（コラム)の名前の一覧
    X1: x軸の列名(columnsの中に含まれる名前)
    Y1: y軸の列名(columnsの中に含まれる名前)
    """
    df = pd.read_csv(file,sep=',',names=columns)
    x,ymean,ystd,ymin,ymax = analyze(df,X1,Y1)
    plt.errorbar(x,ymean,yerr=ystd,fmt='o',color='b',ecolor='gray',capsize=2)
    plt.ylabel(Y1)
    plt.xlabel(X1)
    plt.show()

def plot_2dpcolor(file,columns,X1,X2,Y1):
    df = pd.read_csv(file,sep=',',names=columns)
    x1 = df[X1].values
    x2 = df[X2].values
    y1 = df[Y1].values
    nx1 = len(set(x1))
    nx2 = len(set(x2))
    x1 = x1.reshape(nx1,nx2)
    x2 = x2.reshape(nx1,nx2)
    y1 = y1.reshape(nx1,nx2)
    print(x2)
    plt.pcolor(x1,x2,y1)
    plt.colorbar()
    plt.xlabel(X1)
    plt.ylabel(X2)
    plt.title(Y1)
    plt.show()

def plot_s2dpcolor(file,columns,X1,X2,Y1):
    df = pd.read_csv(file,sep=',',names=columns)
    x1,x2,ymean,ystd,ymin,ymax = analyze2d(df,"id",X1,X2,Y1)
    nx1=len(set(x1))
    nx2=len(set(x2))
    x1=x1.reshape(nx1,nx2)
    x2=x2.reshape(nx1,nx2)
    ymean=ymean.reshape(nx1,nx2)

    plt.pcolor(x1,x2,ymean)
    plt.colorbar()
    plt.xlabel(X1)
    plt.ylabel(X2)
    plt.title(Y1)
    plt.show()
