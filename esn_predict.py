# Copyright (c) 2017-2018 Katori Lab. All Rights Reserved
# NOTE:ESNによる時系列の生成, random spikeによる評価, capacityの評価
# Furuta (2018)にある性能評価指標のcapacityを実装した
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

MM = 100
MM0 = 50
T0 = 5

Nu = 1   #size of input
Nx = 300 #size of dynamical reservior
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
    Wo = np.ones(Ny * (Nx+1)) # Nx+1の１はバイアス項
    Wo = Wo.reshape((Ny, Nx+1))
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
    global X, Y,x
    X = np.zeros((MM, Nx+1))
    Y = np.zeros((MM, Ny))

    n = 0
    x = np.random.uniform(-1, 1, Nx) * 0.2
    xb = np.append(x,1) #バイアス項を追加
    y = np.zeros(Ny)
    X[n, :] = xb
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
        xb = np.append(x,1) #バイアス項を追加
        y = fy(Wo@xb)

        X[n + 1, :] = xb
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
    E = np.identity(Nx+1)
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



def train():
    global D,Ds,Dp,U,Us,Up,Rs,R2s,MM
    global RMSE1,RMSE2,capacity

    #データの準備
    d = np.zeros(MM1+MM2)
    d[delay:]=u[:len(u)-delay] # 入力の時間遅れ時系列を出力とする

    U = np.zeros((MM1+MM2,1))
    D = np.zeros((MM1+MM2,1))
    U[:,0] = u
    D[:,0] = d

    D1 = D[0:MM1]
    U1 = U[0:MM1]

    ### training
    #print("training...")
    MM=MM1
    Dp = np.tanh(D1)
    Up = np.tanh(U1)
    train_network()

def predict():
    global Y,X ,x
    Y = np.zeros((MM2,Ny))
    u = np.zeros((Nu,1))
    sum = np.zeros((Nx,1))
    X = np.zeros((MM, Nx+1))
    x = x.reshape((Nx,1))
    xb = np.append(x,1) #バイアス項を追加
    X[0, :] = xb
    for n in range(MM2-1):
        
        sum += Wi@u
        sum += Wr@x
        #if mode == 0:
        #    sum += Wb@y
        #if mode == 1:  # teacher forcing
        #    d = D[n, :]
        #    sum += Wb@d
        #x = x + 1.0 / tau * (-alpha0 * x + fx(sum))
        x = fx(sum)
        xb = np.append(x,1) #バイアス項を追加
        y = fy(Wo@xb)
        u = [y]
        X[n + 1, :] = xb
        Y[n + 1, :] = y




    return 0





def execute():
    global D,Ds,Dp,U,Us,Up,Rs,R2s,MM,Y
    global RMSE1,RMSE2
    global delay,u,MM1,MM2
    global capacity

    if seed>=0:
        np.random.seed(seed)
    generate_weight_matrix()

    ### generate data
    MM1=2000 # length of training data
    MM2=2000 # length of test data
    u = generate_random_spike1(MM1+MM2)
    d = np.zeros(MM1+MM2)

    delay = 0
    train()
    plot2()

    predict()

    Y = Y[200:]
    Dp = fy(D[200+MM1:])

    ### evaluation
    sum=0
    for j in range(MM0,Y.size):
        sum += (Y[j] - Dp[j])**2
    SUM=np.sum(sum)
    rmse = np.sqrt(SUM/Ny/(MM-MM0))

    RMSE1 = rmse/np.var(Dp)
    RMSE2 = 0
    print(RMSE1)

    if display :
        plot2()

if __name__ == "__main__":
    #config()
    execute()
    #output()
