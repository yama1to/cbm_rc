# Copyright (c) 2018 Katori lab. All Rights Reserved
# NOTE: サーチツール

import os
import subprocess
import sys
import numpy as np
import pandas as pd
import itertools
import datetime
import time
from pprint import pprint
import matplotlib as mpl
mpl.use('Agg')# リモート・サーバ上で図を出力するための設定
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
#from file_report import *
parallel=0
exe=""
file_md = ""
columns=[]
dir_path = os.getcwd() # 結果を出力するディレクトリのパス，デフォルトはカレントディレクトリ

# ディレクトリ指定について
def set_path(path):
    """
    # 結果を出力するディレクトリのパスを指定する
    """
    global dir_path

    # ディレクトリの存在有無の確認
    if os.path.isdir(path): # ディレクトリの存在の確認
        print("Exist Directory: %s" % (path))
        pass
    else:
        print("Create Directory: %s" % (path))
        os.makedirs(path) # ディレクトリの作成

    dir_path = path

### レポートツール TODO:別のファイルに移動
def report(str):
    print(str)
    file_md_path = os.path.join(dir_path, file_md)
    if file_md_path != "":
        f=open(file_md_path,"a")
        f.write(str+"")
        f.close()

def string_now():
    t1=datetime.datetime.now()
    s=t1.strftime('%Y/%m/%d %H:%M:%S')
    return s

def plt_output(fig):
    if fig == '':
        plt.show()
    else:
        fig_path = os.path.join(dir_path, fig)
        plt.savefig(fig_path)
    report_fig(fig)

def report_fig(fig):
    file_md_path = os.path.join(dir_path, file_md)
    if file_md_path != "" and fig !="" :
        f=open(file_md_path,"a")
        f.write("Figure:** %s **  \n" % fig)
        f.write("![](%s)  \n" % (fig))
        f.close()

def execute1(exe,file_csv="",file_fig1="",file_fig2="",file_fig3=""):

    s = "### Execute1 \n"
    report(s)

    e = exe + "display=1 "
    #file_csv="tmp_exe1.csv"
    if file_csv != '':
        file_csv_path = os.path.join(dir_path, file_csv)
        e += "file_csv=%s " % (file_csv_path)

    if file_fig1 != '':
        report_fig(file_fig1)
        file_fig1_path = os.path.join(dir_path, file_fig1)
        e += "file_fig1=%s " % (file_fig1_path)
    if file_fig2 != '':
        report_fig(file_fig2)
        file_fig2_path = os.path.join(dir_path, file_fig2)
        e += "file_fig2=%s " % (file_fig2_path)
    if file_fig3 != '':
        report_fig(file_fig3)
        e += "file_fig3=%s " % (file_fig3_path)

    #print("[execute1]",e)
    subprocess.call(e.split())

    df = pd.read_csv(file_csv_path,sep=',',names=columns)
    x = df.iloc[0]

    s = ""
    s+="Configuration:  \n"
    #s += "Base configuration: `%s`  \n" % exe
    s+="```\n"
    for i,c in enumerate(columns):
        s += "%s: %s\n" % (columns[i],str(x[i]))
    s+="```\n"
    report(s)

def execute(commands,run=1,list=0):# コマンドのリストを受け取り並列実行する
    """
    commandのリストを受け取り、並列実行する。gnu parallelは不使用。
    """
    global parallel
    global file_sh_path
    file_sh="cmd.sh"
    file_sh_path = os.path.join(dir_path, file_sh)

    #print("len:",len(commands))
    #pprint(commands)
    if list==1:
        pprint(commands)
    if run==1:
        n = parallel
        if parallel == 0:
            n=1

        commands = [commands[idx:idx + n] for idx in range(0,len(commands), n)]
        # コマンドをn個に分割して、新たにリストを作る。
        #print(len(commands))
        #print(commands)
        report("Start:" + string_now()+"  \n")
        for cmd in commands:
            f=open(file_sh_path,"w")
            for c in cmd:
                f.write(c+"&\n")
            f.write("wait\n")
            f.close()
            command="sh ./%s" % file_sh_path
            subprocess.call(command.split())

        report("Done :" + string_now()+"  \n")

    if os.path.exists(file_sh_path):
        os.remove(file_sh_path)

def execute2(commands,run=1,list=0):# コマンドのリストを受け取り実行する
    #print("len:",len(commands))
    #pprint(commands)
    if list==1:
        pprint(commands)
    if run==1:
        filename_sh="cmd.sh"
        f=open(filename_sh,"w")
        for c in commands:
            f.write(c+"&\n")
        f.write("wait\n")
        f.close()
        command="sh ./%s" % filename_sh

        #command="parallel -a %s" % filename_sh
        report("Start:" + string_now()+"  \n")
        subprocess.call(command.split())
        report("Done :" + string_now()+"  \n")

def execute0(commands,run=1,list=0):# コマンドのリストを受け取り、parallel で実行する.
    if list==1:
        pprint(commands)
    if run==1:
        filename_sh="cmd.sh"
        f=open(filename_sh,"w")
        for command in commands:
            f.write(command+"\n")
        f.close()
        command="parallel -a %s" % filename_sh
        report("Start:" + string_now()+"  \n")
        subprocess.call(command.split())
        report("Done :" + string_now()+"  \n")

def load_dataframe(file_csv,path=None):
    if path == None:
        file_csv_path = os.path.join(dir_path, file_csv)
    else:
        file_csv_path = os.path.join(path, file_csv)
    df = pd.read_csv(file_csv_path, sep=",", names=columns)
    return df

### Grid scan

def scan1d(file_csv,X1,min=0,max=1,num=11,run=1,list=0):#１変数のスキャン
    s = "### Grid search (scan1d) \n"
    s += "1D grid search on *** %s (min=%f max=%f num=%d) ***  \n" % (X1,min,max,num)
    s += "Base configuration: `%s`  \n" % exe
    s += "Data:**%s**  \n" % (file_csv)
    report(s)

    file_csv_path = os.path.join(dir_path, file_csv)
    c=[]
    id=0
    for x1 in np.linspace(min,max,num):
        c.append(exe+" file_csv=%s id=%d %s=%f " % (file_csv_path,id,X1,x1))
        id+=1
    execute(c,run,list)
    return c

def scan1ds(file_csv,X1,min=0,max=1,num=11,samples=10,run=1,list=0):# seedと１変数のスキャン
    s = "### Grid search (scan1ds) \n"
    s += "1D grid search on *** %s (min=%f max=%f num=%d samples=%d) ***  \n" % (X1,min,max,num,samples)
    s += "Base configuration: `%s`  \n" % exe
    s += "Data:**%s**  \n" % (file_csv)
    report(s)

    file_csv_path = os.path.join(dir_path, file_csv)
    c=[]
    for seed in np.arange(samples):
        id=0
        for x1 in np.linspace(min,max,num):
            c.append(exe+" file_csv=%s seed=%d id=%d %s=%f " % (file_csv_path,seed,id,X1,x1))
            id+=1
    execute(c,run,list)
    return c

def scan2d(file_csv,X1,X2,min1=0.0,max1=1.0,num1=11,min2=0.0,max2=1.0,num2=11,run=1,list=0):#２変数によるスキャン
    s = "### Grid search (scan2d) \n"
    s += "2D grid search on *** %s (min=%f max=%f num=%d) and %s (min=%f max=%f num=%d) ***  \n"\
     % (X1,min1,max1,num1,X2,min2,max2,num2)
    s += "Base configuration: `%s`  \n" % exe
    s += "Data:**%s**  \n" % (file_csv)
    report(s)

    file_csv_path = os.path.join(dir_path, file_csv)
    c=[]
    id=0
    for x1 in np.linspace(min1,max1,num1):
        for x2 in np.linspace(min2,max2,num2):
            c.append(exe+" file_csv=%s id=%d %s=%f %s=%f" % (file_csv_path,id,X1,x1,X2,x2))
            id+=1
    execute(c,run,list)
    return c

def scan2ds(file_csv,X1,X2,min1=0.0,max1=1.0,num1=11,min2=0.0,max2=1.0,num2=11,samples=10,run=1,list=0):#　seedと２変数によるスキャン
    s = "### Grid search (scan2ds) \n"
    s += "2D grid search on *** %s (min=%f max=%f num=%d) and %s (min=%f max=%f num=%d) ***  \n"\
     % (X1,min1,max1,num1,X2,min2,max2,num2)
    s += "Base configuration: `%s`  \n" % exe
    s += "Data:**%s**  \n" % (file_csv)
    report(s)

    file_csv_path = os.path.join(dir_path, file_csv)
    c=[]
    for seed in np.arange(samples):
        id=0
        for x1 in np.linspace(min1,max1,num1):
            for x2 in np.linspace(min2,max2,num2):
                c.append(exe+"file_csv=%s seed=%d id=%d %s=%f %s=%f" % (file_csv_path,seed,id,X1,x1,X2,x2))
                id+=1
    execute(c,run,list)
    return c

### random search

listx=[]
def clear():
    listx=[]

def append(name,min,max):
    listx.append({'type':'f', 'name':name,'min':min, 'max':max})
def appendint(name,min,max):
    listx.append({'type':'i', 'name':name,'min':min, 'max':max})

def random(file_csv,num=1000,run=1,list=0,samples=1):
    s = "### Random search (random) \n"
    s += "%d points search on  \n" % (num)
    for j,cx in enumerate(listx):
        s += "%s: (min:%f max:%f)  \n" % (cx['name'],cx['min'],cx['max'])
    s += "base configuration: `%s`  \n" % exe
    s += "Data: **%s**  \n" % (file_csv)
    report(s)

    file_csv_path = os.path.join(dir_path, file_csv)
    c=[]
    id = 0
    for i in np.arange(num):
        ex1=''
        for j,cx in enumerate(listx):
            if cx['type']=='f' :
                x=np.random.uniform(cx['min'],cx['max'])
                ex1 += "%s=%f " % (cx['name'],x)
            if cx['type']=='i' :
                x=np.random.randint(cx['min'],cx['max'])
                ex1 += "%s=%d " % (cx['name'],x)

        for seed in np.arange(samples):
            ex2="file_csv=%s seed=%d id=%d " % (file_csv_path,seed,id)
            ex = ex2 + ex1
            c.append(exe + ex)

        id+=1

    execute(c,run,list)
    return c

### analyze

def analyze(df0,key,y):
    #df0= pd.read_csv(file_csv,sep=',',names=columns)
    df = df0.groupby(key).aggregate(['mean','std','min','max'])
    x1 = df0.groupby(key)[key].mean().values
    y1mean = df[y,'mean'].values
    y1std  = df[y,'std'].values
    y1min  = df[y,'min'].values
    y1max  = df[y,'max'].values
    return x1,y1mean,y1std,y1min,y1max

def analyze2d(df0,key,X1,X2,Y):
    #df0= pd.read_csv(file_csv,sep=',',names=columns)
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

### plot

def plot1d(file_csv,X1,Y1,fig=''):
    """
    1次元プロットを作成する。
    file_csv: データが含まれるCSVファイル
    X1: x軸の列名(columnsの中に含まれる名前)
    Y1: y軸の列名(columnsの中に含まれる名前)
    """
    df = load_dataframe(file_csv)
    plt.figure()
    plt.plot(df[X1],df[Y1],'o')
    plt.ylabel(Y1)
    plt.xlabel(X1)
    #plt.show()
    plt_output(fig)

def plot1ds(file_csv,X1,Y1,fig=''):
    """
    1次元プロットを作成する。異なるseedのデータについては平均を標準偏差をとる
    file_csv: データが含まれるCSVファイル
    columns: CSVファイルの列（コラム)の名前の一覧
    X1: x軸の列名(columnsの中に含まれる名前)
    Y1: y軸の列名(columnsの中に含まれる名前)
    """
    df = load_dataframe(file_csv)
    x,ymean,ystd,ymin,ymax = analyze(df,X1,Y1)
    plt.figure()
    plt.errorbar(x,ymean,yerr=ystd,fmt='o',color='b',ecolor='gray',capsize=2)
    plt.ylabel(Y1)
    plt.xlabel(X1)
    plt_output(fig)

def plot2d(file_csv,X1,X2,Y1,fig=''):
    df = load_dataframe(file_csv)
    df = df.sort_values('id',ascending=True)
    x1 = df[X1].values
    x2 = df[X2].values
    y1 = df[Y1].values
    nx1 = len(set(x1))
    nx2 = len(set(x2))

    plt.figure()
    for cx2 in set(x2):
        df2 = df[(df[X2]==cx2)]
        x1 = df2[X1].values
        y1 = df2[Y1].values
        plt.plot(x1,y1)

    plt.xlabel(X1)
    plt.ylabel(Y1)
    #plt.show()
    plt_output(fig)

def plot2ds(file_csv,X1,X2,Y1,fig=''):
    df = load_dataframe(file_csv)
    df = df.sort_values('id',ascending=True)
    x1 = df[X1].values
    x2 = df[X2].values
    y1 = df[Y1].values
    nx1 = len(set(x1))
    nx2 = len(set(x2))

    plt.figure()
    for cx2 in set(x2):
        df2 = df[(df[X2]==cx2)]
        x,ymean,ystd,ymin,ymax = analyze(df2,X1,Y1)
        plt.errorbar(x,ymean,yerr=ystd,fmt='o',color='b',ecolor='gray',capsize=2)

    plt.xlabel(X1)
    plt.ylabel(Y1)
    #plt.show()
    plt_output(fig)

def plot2d_pcolor(file_csv,X1,X2,Y1,fig=''):
    df = load_dataframe(file_csv)
    df = df.sort_values('id',ascending=True)
    x1 = df[X1].values
    x2 = df[X2].values
    y1 = df[Y1].values
    nx1 = len(set(x1))
    nx2 = len(set(x2))
    x1 = x1.reshape(nx1,nx2)
    x2 = x2.reshape(nx1,nx2)
    y1 = y1.reshape(nx1,nx2)

    #print(nx1,nx2)
    #print(x2)
    plt.figure()
    plt.pcolor(x1,x2,y1)
    plt.colorbar()
    plt.xlabel(X1)
    plt.ylabel(X2)
    plt.title(Y1)
    #plt.show()
    plt_output(fig)

def plot2ds_pcolor(file_csv,X1,X2,Y1,fig=''):
    df = load_dataframe(file_csv)
    df = df.sort_values('id',ascending=True)
    x1,x2,ymean,ystd,ymin,ymax = analyze2d(df,"id",X1,X2,Y1)
    nx1=len(set(x1))
    nx2=len(set(x2))
    x1=x1.reshape(nx1,nx2)
    x2=x2.reshape(nx1,nx2)
    ymean=ymean.reshape(nx1,nx2)

    plt.figure()
    plt.pcolor(x1,x2,ymean)
    plt.colorbar()
    plt.xlabel(X1)
    plt.ylabel(X2)
    plt.title(Y1)
    #plt.show()
    plt_output(fig)
