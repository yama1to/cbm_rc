# Copyright (c) 2018 Katori Lab. All Rights Reserved
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
import time

#import matplotlib.pyplot as plt

### report tools TODO:別のファイルに移動

def report(text):
    print(text)
    file_md_path = os.path.join(dir_path, file_md)
    if file_md_path != "":
        f=open(file_md_path,"a")
        f.write(text)
        f.close()

def string_now():
    t1=datetime.datetime.now()
    s=t1.strftime('%Y/%m/%d %H:%M:%S')
    return s

####################################
listx=[] # list of configuration of x
vname=[] # vector of names of x and y
columns=[]
xbase=[]
#xbest=[]


parallel=0
exe=""
opt=""
exe_best=""
file_md = "" # markdown file
file_opt="opt.csv"
file_tmp="tmp.csv"
file_sh ="tmp.sh"
dir_path = os.getcwd()

#is_file=0
separator=','
id=0

# パスの指定
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

def list():
    for x in listx:
        print(x)

def clear():
    listx.clear()
#    is_file=0

def appendid():
    listx.append({'type':"id",'name':"id",'value':0})

def append(name,value=0,min=-1,max=1,round=8):
    listx.append({'type':"f", 'name':name, 'value':value, 'min':min, 'max':max, 'variable':1,'round':round})

#def appendxi(name,value,min,max):
#    listx.append({'type':"i", 'name':name, 'value':value, 'min':min, 'max':max, 'variable':1})

def appendseed(name="seed"):
    listx.append({'type':"seed",'name':name,'value':0})

def config():
    vname.clear()
    for cx in listx: vname.append(cx['name'])
    xbase.clear()
    for cx in listx: xbase.append(cx['value'])
    #print(vname)
    #print(xbase)

def execute(commands,run=1,list=0):# コマンドのリストを受け取り並列実行する
    """
    commandのリストを受け取り、並列実行する。gnu parallelは不使用。
    """
    global parallel
    global file_sh_path
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
        
        #report("Start:" + string_now()+"  \n")
        for cmd in commands:
            #file_sh_path="cmd.sh"
            f=open(file_sh_path,"w")
            for c in cmd:
                f.write(c+"&\n")
            f.write("wait\n")
            f.close()
            command="sh ./%s" % file_sh_path
            subprocess.call(command.split())

        #report("Done :" + string_now()+"  \n")

def execute_df(df,op):
    ### Error message
    if len(df.columns) != len(listx):
        print("Error: size of given dataframe does not match listx. dataframe has %d columns, and listx has %d elements" % (len(df.columns),len(listx)) )
        return 1

    if os.path.exists(file_tmp_path):
        os.remove(file_tmp_path)

    ### generate list of commands
    commands = []
    for i in range(len(df.index)):
        opt=" "
        #if is_file == 1:
        opt += "file_csv=" + file_tmp_path + " "

        for j in range(len(listx)):
            if listx[j]['type']=='id':
                value = "%d" % df.ix[i,j]
            if listx[j]['type']=='seed':
                value = "%d" % df.ix[i,j]
            if listx[j]['type']=='f':
                value = "%.8f" % df.ix[i,j]
            if listx[j]['type']=='i':
                value = "%d" % df.ix[i,j]

            opt += df.columns[j] + "=" + str(value) + " "
        command = exe + opt
        commands.append(command)

    #print(commands)
    ### execute list of commands
    execute(commands)
    #if parallel==0:
    #    for command in commands:
    #        print(command)
    #        subprocess.call(command.split())
    #if parallel>=1:
    #    f=open(file_sh_path,"w")
    #    for command in commands: f.write(command+"\n")
    #    f.close()
    #    #TODO: specify num of available cores
    #    #command="parallel -j %d -a %s" % parallel, fcmd
    #    command="parallel -a %s" % file_sh_path
    #    #print(command)
    #    subprocess.call(command.split())

    df2 = pd.read_csv(file_tmp_path,sep=separator,names=columns)

    ### find last(largest) id number in report
    #last_id=0
    #if os.path.exists(file_opt):
    #    df4 = pd.read_csv(file_opt,sep=',',names=vname)
    #    last_id = df4['id'].max()
    #    print("last id:",last_id)

    df2.to_csv(file_opt_path, index=False, mode='a', header=False)
    return df2

def count_key_value(list,key,value):
    count=0
    for c in list:
        if c[key]==value:
            count+=1
    return count

def func1(row):
    return row['x1']+row['x2']

def optimize(op,mm):

    #assert x < 100, "x should be less than 100."

    ### report
    text="### Optimization \n"
    text+="Configuration:  \n"
    text+="```\n"
    for cx in listx:
        text += "{:8s}:{:9.6f}\n".format(cx['name'],cx['value'])
    if "target" in op:
        text += "target: %s \n" % op['target']
    if "TARGET" in op:
        text += "TARGET: %s \n" % op['TARGET']
    if "iteration" in op:
        text += "iteration: %s \n" % op['iteration']
    if "population" in op:
        text += "population: %s \n" % op['population']
    if "samples" in op:
        text += "samples: %s \n" % op['samples']

    #text += "target=%s \n" % op['target']
    text += "```\n"
    text+= "Start:" + string_now() + "  \n"
    report(text)

    ### config
    vname.clear()
    for cx in listx: vname.append(cx['name'])
    xbase.clear()
    for cx in listx: xbase.append(cx['value'])

    ### prepare tmp files
    global file_tmp,file_sh
    t1=datetime.datetime.now()
    file_tmp = "tmp"+str(t1.microsecond) + ".csv"
    file_sh  = "tmp"+str(t1.microsecond) + ".sh"

    ### ディレクトリパス付きの準備
    global file_tmp_path,file_sh_path,file_opt_path
    file_tmp_path = os.path.join(dir_path, file_tmp)
    file_sh_path = os.path.join(dir_path, file_sh)
    file_opt_path = os.path.join(dir_path, file_opt)

    global xbest
    ### Initial options
    target="none"
    target_function="none"
    num_population=10
    num_iteration=10
    num_samples=1
    num_round_default=10
    minmax=0
    operation_id="OPT"

    #np.random.seed(0)

    ### parse options
    if mm=="minimize":
        minmax=-1
        operation_id="MIN"
    if mm=="maximize":
        minmax=+1
        operation_id="MAX"
    if "target" in op: target=op['target']
    if "TARGET" in op: target_function = op['TARGET']
    if "iteration" in op: num_iteration = op['iteration']
    if "population" in op: num_population = op['population']
    if "samples" in op: num_samples = op['samples']


    ### TODO error message
    if target == "none" and target_function == "none" :
        print("Error: target/TARGET not specified"); return 1

    count_seed=count_key_value(listx,'type','seed')
    if count_seed==0 and num_samples>1: print("Error: seed not specified"); return 1

    count_id=count_key_value(listx,'type','id')
    if count_id==0: print("Error: id not appended"); return 1

    ### initialize series and dataframe
    xnames=[]
    for cx in listx:
        xnames.append(cx['name'])
    vx=[]
    for cx in listx:
        vx.append(cx['value'])
    s0 = pd.Series(vx, index=xnames)

    ### prepare dataframe (df0) with random values
    df0 = pd.DataFrame(index=[],columns=xnames)
    #print(df0.columns)
    id = 0
    for i in range(num_population):
        s1 = s0
        #for j in range(len(listx)):
        for j,cx in enumerate(listx):
            #cx=listx[j]
            if cx['type']=='id' :
                s1[j] = id

            if cx['type']=='f' and cx['variable'] :
                x=np.random.uniform(cx['min'],cx['max'])
                if "round" in cx: x = round(x,cx['round'])
                s1[j] = x

            if cx['type']=='i' and cx['variable'] :
                x=np.random.randint(cx['min'],cx['max']+1)
                s1[j] = x

        df0 = df0.append(s1,ignore_index=True)
        id = id + 1

    ### overwrite first line of df0 with xbase
    if 1 and len(xbase)==len(listx): # TODO option to switch the overwriting
        for j in range(len(listx)):
            if listx[j]['type']=='f' and listx[j]['variable'] :
                df0.iat[0,j] = xbase[j]

    #print("df0:")
    #print(df0)

    num_shrink = int(num_population/2)
    num_reflect = num_population - num_shrink
    tbest = -1e99 # larger is better
    sbest = df0.iloc[0]

    ### print configuration

    for cx in listx:
        print(cx)

    im=0
    while im < num_iteration:
        print("Iteration:",im+1,"/",num_iteration)
        ### df1
        if im==0 : # for the first iteration, set df1 random values(df0)
            df1 = df0
        else:
            df1previous = df1
            df1 = pd.DataFrame(index=[],columns=xnames)
            id=0
            for k in range(num_shrink):# crossover
                s1 = s0
                #s1[0] = id
                for j,cx in enumerate(listx):
                    if cx['type']=='id':
                        s1[j] = id
                    if cx['type']=='f' and cx['variable']:
                        x = (sbest[j] + df1previous.iloc[k+1,j])/2.0
                        if "round" in cx: x = round(x,cx['round'])
                        if x>cx['max']: x=cx['max']
                        if x<cx['min']: x=cx['min']
                        s1[j] = x
                    if cx['type']=='i' and cx['variable'] :
                        x = int( (sbest[j] + df1previous.iloc[k+1,j])/2.0 )
                        s1[j] = x
                df1 = df1.append(s1,ignore_index=True)
                id=id+1
            for k in range(num_reflect):# mutation
                s1 = s0
                for j,cx in enumerate(listx):
                    cx=listx[j]
                    if cx['type']=='id':
                        s1[j] = id
                    if cx['type']=='f' and cx['variable']:
                        x = sbest[j] + (sbest[j] - df1previous.iloc[k+1,j])*1.5
                        if "round" in cx: x = round(x,cx['round'])
                        if x>cx['max']: x=cx['max']
                        if x<cx['min']: x=cx['min']
                        s1[j] = x
                    if cx['type']=='i' and cx['variable'] :
                        x = int( sbest[j] + (sbest[j] - df1previous.iloc[k+1,j])*1.5 )
                        s1[j] = x
                df1 = df1.append(s1,ignore_index=True)
                id=id+1
        #print("df1:")
        #print(df1)

        ### df2: multiply df1 with different seeds
        df2 = pd.DataFrame(index=[],columns=xnames)
        for i in range(num_population):
            s1 = df1.iloc[i]
            #print(s1)
            for i_sample in range(num_samples):
                s2 = s1
                for j in range(len(listx)):
                    if listx[j]['type']=='seed':
                        s2[j] = i_sample
                #print(s2)
                df2 = df2.append(s2,ignore_index=True)
        #print("df2:")
        #print(df2)

        ### execute
        op2={'operation_id':operation_id}
        df3 = execute_df(df2,op2)

        ### 最適化の評価値(TARGET)の計算とソート
        if target != "none":
            df3['TARGET'] = df3[target]
        if target_function != "none":
            df3['TARGET']=df3.apply(target_function,axis=1)
        if minmax==+1:#maximize
            df3 = df3.sort_values('TARGET',ascending=False)
        if minmax==-1:#minimize
            df3 = df3.sort_values('TARGET',ascending=True)
        #print("df3:")
        print(df3)

        ### id による集約（異なるseed値について平均をとる）とソート
        df4=df3.groupby(df3['id']).mean()
        if minmax==+1:#maximize
            df4 = df4.sort_values('TARGET',ascending=False)
        if minmax==-1:#minimize
            df4 = df4.sort_values('TARGET',ascending=True)
        i0 = df4.index[0]
        t0 = df4['TARGET'].iloc[0]

        #print("df4:\n",df4)
        #print("df4.index[0]",df4.index[0],"　　df4[target]",df4[target].iloc[0])
        #print("im: ",im,"i0: ",i0,"t0: ",t0,"tbest:",tbest)

        #print(df4.iloc[0])
        #print(df1.iloc[i0])

        if num_samples>1:
            print(df4)

        #xbest = df1.iloc[i0] # current best
        #c2best = df4.iloc[0] # current best
        #print(xbest)
        #print(c2best)
        #print("asdf:",xbest[0],c2best[0])

        if im == 0:
            tbest = t0
            sbest = df1.iloc[i0]
            xbest = df4.iloc[0]
        else:
            if minmax == +1 and t0 > tbest:
                tbest = t0
                sbest = df1.iloc[i0]
                xbest = df4.iloc[0]
            if minmax == -1 and t0 < tbest:
                tbest = t0
                sbest = df1.iloc[i0]
                xbest = df4.iloc[0]
            #print("best:",tbest,i0)
            #print(sbest)

        im+=1

    ### exe best
    opt = " "
    value = ""
    name = ""
    for i in range(len(listx)):
        if listx[i]['type']=='f':
            name = listx[i]['name']
            value = "%.8f" % xbest[name]
            opt += listx[i]['name'] + "=" + str(value) + " "
        if listx[j]['type']=='i':
            name = listx[i]['name']
            value = "%d" % xbest[name]
            opt += listx[i]['name'] + "=" + str(value) + " "
        #print("debug ",i,listx[i]['name'],value)

    global exe_best
    exe_best = exe + opt

    ### print result
    report("Done :" + string_now() + "  \n")
    text="Result:  \n"
    text+="```\n"
    for i in range(len(xbest)):
        #text += "{:8s}:{:9.6f}\n".format(df4.columns[i],df4.iloc[0,i])
        text += "{:8s}:{:9.6f}\n".format(df4.columns[i],xbest[i])
    text += "```\n"
    text += "best:  \n"
    text += "```\n"
    text += exe_best + "\n"
    text += "```\n"
    report(text)



    ### clean up
    if os.path.exists(file_tmp_path):
        os.remove(file_tmp_path)
    if os.path.exists(file_sh_path):
        os.remove(file_sh_path)

    #for j,cx in enumerate(listx):
    #    listx[j]['value'] = sbest[j]
    #for j,cx in enumerate(listx):
    #    if cx['type']!='id' and cx['type']!='seed':
    #        print(cx['name'],cx['value'])

def maximize(op):
    optimize(op,"maximize")
def minimize(op):
    optimize(op,"minimize")
