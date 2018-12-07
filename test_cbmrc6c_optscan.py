# Copyright (c) 2018 Katori lab. All Rights Reserved
# NOTE:

import numpy as np
import itertools
import subprocess
import sys
from pprint import pprint
import optimization as opt

opt.parallel=40
opt.exe="python cbmrc6c.py display=0 ex=opt1"
opt.filename_tmp="data_cbmrc6c_opt1.csv"
opt.filename_rec="data_cbmrc6c_opt1_rec.csv"

opt.clear()
opt.appendid()
opt.appendseed("seed")# 乱数のシード
opt.appendxfr("alpha_r",0.2,0,1,2)
opt.appendxfr("alpha_b",0.2,0,1,2)
opt.appendxfr("alpha_s",0.2,0,2,2)
opt.config()
opt.names=['seed','id','NN','Nh','alpha_i','alpha_r','alpha_b','alpha_s','alpha0','alpha1','beta_i','beta_r','beta_b','Temp','lambda0','RMSE1','RMSE2','count_gap']

def func(row):
    return row['RMSE1'] + 0.1*row['count_gap']

#opt.minimize({'target':"y1",'iteration':3,'population':10,'samples':5})
#opt.minimize({'TARGET':func,'iteration':10,'population':20,'samples':20})

### scan
#alpha_r = opt.cbest['alpha_r']
#alpha_b = opt.cbest['alpha_b']
#alpha_s = opt.cbest['alpha_s']
alpha_r = 0.55
alpha_b = 0.16
alpha_s = 1.36

id=0
exe="python cbmrc6c.py display=0 "

def scan1_alpha_r():
    c=[]
    for seed in np.arange(20):
        for alpha_r in np.linspace(0,1,41):
            c.append(exe+"ex=scan1_alpha_r seed=%d id=%d alpha_r=%f alpha_b=%f alpha_s=%f" % (seed,id,alpha_r,alpha_b,alpha_s))
    return c

def scan1_alpha_b():
    c=[]
    for seed in np.arange(20):
        for alpha_b in np.linspace(0,1,41):
            c.append(exe+"ex=scan1_alpha_b seed=%d id=%d alpha_r=%f alpha_b=%f alpha_s=%f" % (seed,id,alpha_r,alpha_b,alpha_s))
    return c

def scan1_alpha_s():
    c=[]
    for seed in np.arange(20):
        for alpha_s in np.linspace(0,2,41):
            c.append(exe+"ex=scan1_alpha_s seed=%d id=%d alpha_r=%f alpha_b=%f alpha_s=%f" % (seed,id,alpha_r,alpha_b,alpha_s))
    return c

def execute(commands):
    filename_sh="cmd.sh"
    f=open(filename_sh,"w")
    for command in commands:
        f.write(command+"\n")
    f.close()
    command="parallel -j 45 -a %s" % filename_sh
    subprocess.call(command.split())

#commands=scan1_alpha_r()
#pprint(commands)
#execute(commands)
#execute(scan1_alpha_r())
#execute(scan1_alpha_b())
execute(scan1_alpha_s())
