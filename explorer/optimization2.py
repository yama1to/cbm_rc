# Copyright (c) 2018-2019 Katori Lab. All Rights Reserved
# Author: Yuichi Katori (yuichi.katori@gmail.com)
# NOTE: optimization.py

import subprocess
import sys
import re
import numpy as np
import copy
import os
import pandas as pd
import datetime
import copy
import scipy.optimize

#import time
from . import common

listx=[] # list of configuration of x
best_config = None
num_samples = None
config_opt = None
target = None
target_function = None
operation = None

def clear():
    listx.clear()

def append(name,value=0,min=-1,max=1,round=8):
    listx.append({'type':"f", 'name':name, 'value':value, 'min':min, 'max':max, 'variable':1,'round':round})

def execute_df(config,df):
    ### Error message
    if len(df.columns) != len(listx)+2:
        print("Error: size of given dataframe does not match listx. dataframe has %d columns, and listx has %d elements" % (len(df.columns),len(listx)) )
        return 1

    #print("df:\n",df)
    ### prepare dataframe
    id=0
    df1 = pd.DataFrame(index=[],columns=common.columns)
    for i in range(len(df.index)):
        cnf = copy.copy(config)
        s0 = df.iloc[i]
        #print(config)
        for j in range(len(df.columns)):
            setattr(cnf,df.columns[j],s0[j])
        s1 = common.config2series(cnf)
        df1 = df1.append(s1,ignore_index=True)

    ## execute
    df1 = common.execute(cnf,df1)
    #print("df1:\n",df1)
    return df1

def count_key_value(list,key,value):
    count=0
    for c in list:
        if c[key]==value:
            count+=1
    return count

def func1(row):
    return row['x1']+row['x2']

def function(x):
    """
    受け取った変数xについて、seedを設定してコードを実行し、targetの平均値を返す。
    id は変数（パラメータ）に対応する。この関数では常に0。
    seed 乱数のシード
    """
    ###
    xnames=[]
    xnames.append("id")
    xnames.append("seed")
    for cx in listx:
        xnames.append(cx['name'])
    #print("xmanes",xnames)

    vx=[]
    vx.append(0)# id
    vx.append(0)# seed
    #for cx in listx:
    for i,cx in enumerate(listx):
        #x.append(cx['value'])
        vx.append(x[i])
    s1 = pd.Series(vx, index=xnames)
    #print("s1\n",s1)

    ### df2: multiply df1 with different seeds
    df2 = pd.DataFrame(index=[],columns=xnames)
    for i_sample in range(num_samples):
        s2 = s1.copy()
        s2[1] = i_sample
        df2 = df2.append(s2,ignore_index=True)
    #print("df2:\n",df2)

    ### execute, df2のパラメータ値についてプログラムを実行
    df3 = execute_df(config_opt,df2)
    #print("df3:\n",df3)

    ### 最適化の評価値(TARGET)の計算とソート
    if target != None:
        df3['TARGET'] = df3[target]
    if target_function != None:
        df3['TARGET']=df3.apply(target_function,axis=1)
    # ソート（不要）
    if operation == "max":#maximize
        df3 = df3.sort_values('TARGET',ascending=False)
    if operation == "min":#minimize
        df3 = df3.sort_values('TARGET',ascending=True)
    #print("df3 sorted\n",df3)

    ### id による集約（異なるseed値について平均をとる）
    df4 = df3.groupby(df3['id']).mean()

    t0 = df4['TARGET'].iloc[0] #
    #print("df4:\n",df4)

    return t0

def maximize(csv=None,config=None,target=None,TARGET=None,population=10,iteration=10,samples=1):
    optimize("max",csv,config,target,TARGET,population,iteration,samples)

def minimize(csv=None,config=None,target=None,TARGET=None,population=10,iteration=10,samples=1):
    optimize("min",csv,config,target,TARGET,population,iteration,samples)


def particle(fun=None,x0=None,bounds=None):
    return 0

def optimize(operation_,csv,config,target_,target_function_,num_population,num_iteration,num_samples):
    """
    この関数の内部では、データフレームを使わない。
    """
    global config_opt
    global operation
    global target
    global target_function

    ### setup
    if config == None:
        config_opt = copy.copy(common.config)
    else:
        config_opt = config
    if csv==None:
        #csv=common.name_file(common.prefix+"_opt.csv")
        filename=common.prefix+"_opt.csv"
        csv=common.name_file(filename,path=None)
    #config.csv=csv

    #filename=common.prefix+"_opt"+str(im)+".csv"
    filename=common.prefix+"_opt.csv"
    csv_tmp=common.name_file(filename,path=None)
    config_opt.csv=csv_tmp

    if hasattr(config_opt,'plot'): setattr(config_opt,'plot',False)
    if hasattr(config_opt,'show'): setattr(config_opt,'show',False)
    if hasattr(config_opt,'savefig'): setattr(config_opt,'savefig',False)

    operation = operation_
    target = target_
    target_function = target_function_

    ### TODO error message
    if target == None and target_function == None :
        print("Error: target/TARGET not specified"); return 1

    ### レポート
    text="### Optimization \n"
    text+="Configuration:  \n"
    text += "operation: %s \n" % operation
    text+="```\n"
    for j,cx in enumerate(listx):
        if cx['type']=='f':
            text += "{:8s}:{:9.6f}[{:9.6f},{:9.6f}]({:d})\n".format(cx['name'],cx['value'],cx['min'],cx['max'],cx['round'])
        #print("xxx",cx['min'])

    if target != None:
        text += "target: %s \n" % target
    if target_function != None:
        text += "TARGET: %s \n" % target_function

    text += "iteration: %s \n" % num_iteration
    text += "population: %s \n" % num_population
    text += "samples: %s \n" % num_samples

    #text += "target=%s \n" % op['target']
    text += "```\n"
    text+= "Start:" + common.string_now() + "  \n"
    common.report(text)


    x0=[]
    bounds=[]
    for cx in listx:
        x0.append(cx['value'])
        bounds.append((cx['min'],cx['max']))

    #res = scipy.optimize.minimize(fun=function,x0=x0,bounds=bounds,method="nelder-mead",options={'maxiter':10})

    ### particle

    num_shrink = int(num_population/2)
    num_reflect = num_population - num_shrink

    ### prepare dataframe (df0) with random values
    ### df0: ランダムに決めた変数を初期値とする。１行目は設定された値

    x0 = np.zeros(0)
    for cx in listx:
        x0 = np.append(x0,cx['value'])

    X0 = x0
    for i in range(num_population-1):
        x = np.zeros(len(x0))
        for j,cx in enumerate(listx):
            x[j] = np.random.uniform(cx['min'],cx['max'])
            if "round" in cx: x[j] = np.round(x[j],cx['round'])
        X0 = np.vstack([X0,x])
    #print("X0:\n",X0)

    ### 最適化のメインループ
    tbest = -1e99 # 暫定最適な返り値、大きな値がより良い
    xbest = X0[0]

    im=0
    while im < num_iteration:
        print("Iteration:",im+1,"/",num_iteration)
        ### df1
        if im==0 : # for the first iteration, set df1 random values(df0)
            X1 = X0
        else:
            #Xprev = copy.copy(X1)
            X1 = np.zeros((0,len(x)))
            for k in range(num_shrink):# crossover
                x = (xbest + X2[1+k])/2.0
                X1 = np.vstack([X1,x])
            for k in range(num_reflect):# mutation
                x = xbest + (xbest - X2[1+k])*1.0
                X1 = np.vstack([X1,x])

            #print("X1",X1)
            for i in range(num_population):
                x = X1[i]
                for j,cx in enumerate(listx):
                    if "round" in cx: x[j] = round(x[j],cx['round'])
                    if x[j] > cx['max']: x[j] = cx['max']
                    if x[j] < cx['min']: x[j] = cx['min']

        Y = np.zeros(0)
        for i,x in enumerate(X1):
            y = function(x)
            Y = np.append(Y,y)
            #print(x,y)

        if operation == "max":
            asY=np.argsort(Y)[::-1]
            sY =np.sort(Y)[::-1]
        if operation == "min":
            asY=np.argsort(Y)
            sY =np.sort(Y)

        X2 = np.zeros_like(X1)
        for i1,i2 in enumerate(asY):
            X2[i1] = X1[i2]

        if im == 0:
            tbest = sY[0]
            xbest = X2[0]
        else:
            if operation == "max" and sY[0] > tbest:
                tbest = sY[0]
                xbest = X2[0]
            if operation == "min" and sY[0] < tbest:
                tbest = sY[0]
                xbest = X2[0]

        print("X1\n",X1)
        print(" Y",Y)
        print("sY",sY,asY)
        print("X2:\n",X2)
        print("xbest",xbest)
        print("tbest",tbest)
        #print("xbest:\n",xbest)
        im+=1
    ### 最適化のメインループここまで

    global best_config
    best_config = config_opt
    for i,cx in enumerate(listx):
        setattr(best_config,cx['name'],xbest[i])

    ### print result
    common.report("Done :" + common.string_now() + "  \n")
    text="Optimization result:  \n"
    text+="```\n"
    for i,cx in enumerate(listx):
        text += "{:8s}:{:9.6f}\n".format(cx['name'],xbest[i])
    text += "terget: %s \n" % tbest
    text += "```\n"
    common.report(text)
