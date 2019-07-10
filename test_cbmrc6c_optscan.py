# Copyright (c) 2018 Katori lab. All Rights Reserved
# NOTE: cbmrc6c

import optimization as opt
from scan import *

opt.parallel=40
opt.filename_tmp="data_cbmrc6c_opt1_tmp.csv"
opt.filename_rec="data_cbmrc6c_opt1_rec.csv"
opt.exe="python cbmrc6c.py display=0 "
columns=['seed','id','NN','Nh','alpha_i','alpha_r','alpha_b','alpha_s','alpha0','alpha1','beta_i','beta_r','beta_b','Temp','lambda0','RMSE1','RMSE2','count_gap']

opt.clear()
opt.appendfile()
opt.appendid()
opt.appendseed("seed")# 乱数のシード
opt.appendxfr("alpha_r",0.2,0,1,2)
opt.appendxfr("alpha_i",0.2,0,1,2)
opt.appendxfr("alpha_s",0.2,0,2,2)
opt.config()
opt.names=columns

def func(row):
    return row['RMSE1'] + 0.1*row['count_gap']

#opt.minimize({'target':"y1",'iteration':3,'population':10,'samples':5})
#opt.minimize({'TARGET':func,'iteration':10,'population':20,'samples':20})

### scan
#exe = "python cbmrc6c.py display=0 alpha_r=0.55 alpha_b=0.16 alpha_s=1.36 "
exe = "python cbmrc6c.py display=0 "
#exe+= "alpha_r=0.55 alpha_b=0.16 alpha_s=1.36 " # alpha_bも含めているがこのパラメータは使用していない。
#exe+= "alpha_r=%f alpha_i=%f alpha_s=%f " % (opt.xbest['alpha_r'],opt.xbest['alpha_i'],opt.xbest['alpha_s'])

f1 = "data2_cbmrc6c_scan1_alpha_r.csv"
c = scan_s1d(exe,f1,20,"alpha_r",0,1,41)
#pprint(c)
#execute(c)

f2 = "data2_cbmrc6c_scan1_alpha_i.csv"
c = scan_s1d(exe,f2,20,"alpha_i",0,1,41)
pprint(c)
#execute(c)

f3 = "data2_cbmrc6c_scan1_alpha_s.csv"
c = scan_s1d(exe,f3,20,"alpha_s",0,2,41)
pprint(c)
#execute(c)

def plot1(filename,key):
    #key='alpha_r'
    df1 = pd.read_csv(filename,sep=',',names=columns)
    print(df1)
    df3 = df1.groupby(key).aggregate(['mean','std'])
    x1 = df1.groupby(key)[key].mean().values
    y1mean = df3['RMSE1','mean'].values
    y1std  = df3['RMSE1','std'].values
    y2mean = df3['count_gap','mean'].values
    y2std  = df3['count_gap','std'].values

    plt.subplot(2,1,1)
    plt.errorbar(x1,y1mean,yerr=y1std,fmt='o',color='b',ecolor='gray',capsize=2)
    plt.ylabel('RMSE1')

    plt.subplot(2,1,2)
    plt.errorbar(x1,y2mean,yerr=y2std,fmt='o',color='b',ecolor='gray',capsize=2)
    plt.ylabel('gap')

    plt.xlabel(key)
    plt.show()

plot1(f1,"alpha_r")
plot1(f2,"alpha_i")
plot1(f3,"alpha_s")
