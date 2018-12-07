# Copyright (c) 2018 Katori lab. All Rights Reserved
# NOTE:


import numpy as np
import itertools
import subprocess
import sys
from pprint import pprint

exe="python cbmrc6b.py display=0 "
alpha_r=0.11
alpha_b=0.66
alpha_s=0.46
id=0

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
        for alpha_s in np.linspace(0,1,41):
            c.append(exe+"ex=scan1_alpha_s seed=%d id=%d alpha_r=%f alpha_b=%f alpha_s=%f" % (seed,id,alpha_r,alpha_b,alpha_s))
    return c

def scan2b():
    c=[]
    X1 = np.linspace(0,1,5)
    X2 = np.linspace(0,1,5)
    for seed in np.arange(2):
        id=0
        for x1,x2 in itertools.product(X1,X2):
            c.append(exe+"ex=scan1 seed=%d id=%d x1=%f x2=%f" % (seed,id,x1,x2))
            id+=1
    return c

def scan3():
    c=[]
    for i in np.arange(30):
        seed = np.random.randint(0,10)
        x1 = np.random.uniform(0,1)
        x2 = np.random.uniform(0,1)
        c.append(exe+"ex=scan1 seed=%d id=%d x1=%f x2=%f" % (seed,id,x1,x2))
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
execute(scan1_alpha_r())
execute(scan1_alpha_b())
execute(scan1_alpha_s())
