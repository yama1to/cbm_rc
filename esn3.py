# Copyright (c) 2017-2018 Katori Lab. All Rights Reserved
# NOTE:ESNによる時系列の生成, random spikeによる評価,
import numpy as np
import scipy.linalg
from numpy.linalg import svd, inv, pinv

import matplotlib as mpl
#mpl.use('Agg')## サーバ上で画像を出力するための設定。ローカルで表示する際はコメントアウトする
import matplotlib.pyplot as plt

import sys
import copy
from arg2x import *
from generate_data_sequence import *

file_csv = "data_esn1.csv"
file_fig1 = "data_esn1_fig1.png"
display = 1

dataset = 4
seed=-1 # 乱数生成のためのシード
id=0

MM = 200
MM0 = 50
T0 = 5

Nu = 1   #size of input
Nx = 100 #size of dynamical reservior
Ny = 1   #size of output

sigma_np = -5
alpha_r = 0.8
alpha_b = 0.0 # 0.8
alpha_i = 0.8
beta_r = 0.1
beta_b = 0.1
beta_i = 0.1
alpha0 = 0.7
tau = 2
lambda0 = 0.1

def config():
    global file_csv,file_fig1,display,dataset,seed,id,Nx,alpha_i,alpha_r,alpha_b,alpha0,tau,beta_i,beta_r,beta_b,lambda0
    args = sys.argv
    for s in args:
        file_csv= arg2a(file_csv,"file_csv=",s)
        file_fig1=arg2a(file_fig1,"file_fig1=",s)
        display = arg2i(display,"display=",s)
        dataset = arg2i(dataset,"dataset=",s)
        seed    = arg2i(seed,"seed=",s)
        id      = arg2i(id,"id=",s)
        Nx      = arg2i(Nx, 'Nx=', s)
        alpha_i = arg2f(alpha_i,"alpha_i=",s)
        alpha_r = arg2f(alpha_r,"alpha_r=",s)
        alpha_b = arg2f(alpha_b,"alpha_b=",s)
        alpha0  = arg2f(alpha0,"alpha0=",s)
        tau     = arg2f(tau,"tau=",s)
        beta_i  = arg2f(beta_i,"beta_i=",s)
        beta_r  = arg2f(beta_r,"beta_r=",s)
        beta_b  = arg2f(beta_b,"beta_b=",s)
        lambda0 = arg2f(lambda0, 'lambda0=', s)

def output():
    str="%d,%d,%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n" \
    % (dataset,seed,id,Nx,alpha_i,alpha_r,alpha_b,alpha0,tau,beta_i,beta_r,beta_b,lambda0,RMSE1,RMSE2,capacity)

    f=open(file_csv,"a")
    f.write(str)
    f.close()

def generate_weight_matrix():
    global Wr, Wb, Wo, Wi
    ### Wr
    Wr0 = np.zeros(Nx * Nx)
    nonzeros = Nx * Nx * beta_r
    Wr0[0:int(nonzeros / 2)] = 1
    Wr0[int(nonzeros / 2):int(nonzeros)] = -1
    np.random.shuffle(Wr0)
    Wr0 = Wr0.reshape((Nx, Nx))
    v = scipy.linalg.eigvals(Wr0)
    lambda_max = max(abs(v))
    Wr = Wr0 / lambda_max * alpha_r

    # print("lamda_max",lambda_max)
    # print("Wr:")
    # print(Wr)

    ### Wb
    Wb = np.zeros(Nx * Ny)
    Wb[0:int(Nx * Ny * beta_b / 2)] = 1
    Wb[int(Nx * Ny * beta_b / 2):int(Nx * Ny * beta_b)] = -1
    np.random.shuffle(Wb)
    Wb = Wb.reshape((Nx, Ny))
    Wb = Wb * alpha_b
    # print("Wb:")
    # print(Wb)

    ### Wi
    Wi = np.zeros(Nx * Nu)
    Wi[0:int(Nx * Nu * beta_i / 2)] = 1
    Wi[int(Nx * Nu * beta_i / 2):int(Nx * Nu * beta_i)] = -1
    np.random.shuffle(Wi)
    Wi = Wi.reshape((Nx, Nu))
    Wi = Wi * alpha_i
    # print("Wi:")
    # print(Wi)

    ### Wo
    Wo = np.ones(Ny * Nx)
    Wo = Wo.reshape((Ny, Nx))
    Wo = Wo
    # print(Wo)


def fx(x):
    return np.tanh(x)

def fy(x):
    return np.tanh(x)

def fyi(x):
    return np.arctanh(x)

def fr(x):
    return np.fmax(0, x)

def run_network(mode):
    global X, Y
    X = np.zeros((MM, Nx))
    Y = np.zeros((MM, Ny))

    n = 0
    x = np.random.uniform(-1, 1, Nx) * 0.2
    y = np.zeros(Ny)
    X[n, :] = x
    Y[n, :] = y
    for n in range(MM - 1):
        sum = np.zeros(Nx)
        u = Up[n, :]
        sum += Wi@u
        sum += Wr@x
        #if mode == 0:
        #    sum += Wb@y
        #if mode == 1:  # teacher forcing
        #    d = D[n, :]
        #    sum += Wb@d
        #x = x + 1.0 / tau * (-alpha0 * x + fx(sum))
        x = fx(sum)
        y = fy(Wo@x)

        X[n + 1, :] = x
        Y[n + 1, :] = y
        # print(y)
        # print(X)

def train_network():
    global Wo

    run_network(1) # run netwrok with teacher forcing

    M = X[T0:, :]
    invD = fyi(Dp)
    G = invD[T0:, :]

    ### Ridge regression
    E = np.identity(Nx)
    TMP1 = inv(M.T@M + lambda0 * E)
    WoT = TMP1@M.T@G
    Wo = WoT.T
    #print("WoT\n", WoT)

def test_network():
    run_network(0)

def plot(data):
    fig, ax = plt.subplots(1,1)
    ax.cla()
    ax.plot(data)
    plt.show()

def plot1():
    fig=plt.figure()
    ax = fig.add_subplot(3,1,1)
    ax.cla()
    ax.plot(Up)
    ax.set_title("Up")

    ax = fig.add_subplot(3,1,2)
    ax.cla()
    ax.plot(Dp)
    ax.set_title("Dp")

    ax = fig.add_subplot(3,1,3)
    ax.cla()
    ax.plot(Y)
    ax.set_title("Y")

    plt.show()
    plt.savefig(file_fig1)

def plot2():
    fig=plt.figure(figsize=(20, 12))
    Nr=3
    t1=100
    ax = fig.add_subplot(Nr,1,1)
    ax.cla()
    ax.set_title("U")
    ax.plot(Up[t1:])

    ax = fig.add_subplot(Nr,1,2)
    ax.cla()
    ax.set_title("X")
    ax.plot(X[t1:,:100])

    ax = fig.add_subplot(Nr,1,3)
    ax.cla()
    ax.set_title("Y, Ytarget")
    ax.set_ylim(-1,1)
    ax.plot(Y[t1:])
    ax.plot(Dp[t1:],'--')

    plt.show()
    plt.savefig(file_fig1)

def execute():
    global D,Ds,Dp,U,Us,Up,Rs,R2s,MM
    global RMSE1,RMSE2,capacity
    if seed>=0:
        np.random.seed(seed)
    generate_weight_matrix()

    ### generate data
    if dataset==1:
        MM1=300 # length of training data
        MM2=400 # length of test data
        D, U = generate_simple_sinusoidal(MM1+MM2)
    if dataset==2:
        MM1=300 # length of training data
        MM2=300 # length of test data
        D, U = generate_complex_sinusoidal(MM1+MM2)
    if dataset==3:
        MM1=1000 # length of training data
        MM2=1000 # length of test data
        D, U = generate_coupled_lorentz(MM1+MM2)
    if dataset==4:
        MM1=1000 # length of training data
        MM2=1000 # length of test data
        D, U = generate_random_spike(MM1+MM2,Nu=1)

    D1 = D[0:MM1]
    U1 = U[0:MM1]
    D2 = D[MM1:MM1+MM2]
    U2 = U[MM1:MM1+MM2]

    ### training
    #print("training...")
    MM=MM1
    Dp = np.tanh(D1)
    Up = np.tanh(U1)
    train_network()

    ### test
    #print("test...")
    MM=MM2
    Dp = np.tanh(D2)
    Up = np.tanh(U2)
    test_network()

    ### evaluation
    sum=0
    for j in range(MM0,MM):
        sum += (Y[j] - Dp[j])**2
    SUM=np.sum(sum)
    RMSE1 = np.sqrt(SUM/Ny/(MM-MM0))
    RMSE2 = 0
    print("RMSE1:",RMSE1)

    ### capacity
    y1=Y[:,0]
    y2=Dp[:,0]
    sum=0
    for delay in range(1,10):
        y1b=y1[:len(y1)-delay]
        y2b=y2[delay:]
        cor=np.corrcoef(y1b,y2b)
        cor=cor[0,1]
        cor2=cor**2
        sum+=cor2
        #print(cor2)
        #print(delay,cor2)
    capacity = sum
    print("capacity:",capacity)

    if display :
        plot2()

if __name__ == "__main__":
    config()
    execute()
    output()
