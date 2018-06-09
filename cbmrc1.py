# Copyright (c) 2017-2018 Katori Lab. All Rights Reserved
# Author: Yuichi Katori (yuichi.katori@gmail.com)
# DONE: 基本的なESNを実装する。クラスは使わずに。

import numpy as np
import scipy.linalg
from numpy.linalg import svd, inv, pinv
import matplotlib.pyplot as plt

T1 = 200
T0 = 5

Nu = 2   #size of input
Nx = 100 #size of dynamical reservior
Ny = 2   #size of output

NN=100
MM=200

sigma_np = -5
alpha_r = 0.8
alpha_b = 0.8
alpha_i = 0.8
beta_r = 0.1
beta_b = 0.1
beta_i = 0.1
alpha0 = 0.7
tau = 2
lambda0 = 0.1

def generate_data_sequence1():
    D = np.zeros((T1, Ny))
    U = np.zeros((T1, Nu))
    cy = np.linspace(0, 1, Ny)
    cu = np.linspace(0, 1, Nu)
    for n in range(T1):
        t = 0.1 * n
        d = np.sin(t + cy) * 0.8
        # d=np.sin(t+c)*np.exp(-0.1*(t-10)**2)*0.5
        u = 0.0 * cu
        D[n, :] = d
        U[n, :] = u
    return (D, U)

def generate_data_sequence2():
    D = np.zeros((T1, Ny))
    U = np.zeros((T1, Nu))
    cy = np.linspace(0, 1, Ny)
    cu = np.linspace(0, 1, Nu)
    for n in range(T1):
        t = 0.1 * n
        d = np.sin(t + cy) * 0.8
        # d=np.sin(t+c)*np.exp(-0.1*(t-10)**2)*0.5
        u = np.sin(t*0.3 + cu) * 0.8
        D[n, :] = d
        U[n, :] = u
    return (D, U)

def generate_s_sequence():
    Ds0 = np.zeros((MM, Ny, NN))
    Us0 = np.zeros((MM, Nu, NN))
    for m in range(MM):
        for p in range(2):
            Ds0[m][p][0:int(NN*Dp[m][p])] = 1
            Ds0[m][p][int(NN*Dp[m][p]):int(NN)] = 0
            Us0[m][p][0:int(NN*Up[m][p])] = 1
            Us0[m][p][int(NN*Up[m][p]):int(NN)] = 0
    Ds = np.reshape(Ds0, (Ny, MM*NN))
    Us = np.reshape(Us0, (Nu, MM*NN))
    return (Us)

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

def fsgm(x):
    return 1.0/(1.0+np.exp(-x))

def run_network(mode):
    global X, Y
    X = np.zeros((T1, Nx))
    Y = np.zeros((T1, Ny))

    n = 0
    x = np.random.uniform(-1, 1, Nx) * 0.2
    y = np.zeros(Ny)
    X[n, :] = x
    Y[n, :] = y
    for n in range(T1 - 1):
        sum = np.zeros(Nx)
        u = U[n, :]
        sum += Wi@u
        sum += Wr@x
        if mode == 0:
            sum += Wb@y
        if mode == 1:  # teacher forcing
            d = D[n, :]
            sum += Wb@d
        x = x + 1.0 / tau * (-alpha0 * x + fx(sum))
        y = fy(Wo@x)

        X[n + 1, :] = x
        Y[n + 1, :] = y
        # print(y)
        # print(X)

def train_network():
    global Wo

    run_network(1) # run netwrok with teacher forcing

    M = X[T0:, :]
    invD = fyi(D)
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

def plot2():
    fig=plt.figure()
    ax1 = fig.add_subplot(3,1,1)
    ax1.cla()
    ax1.plot(U)
    ax2 = fig.add_subplot(3,1,2)
    ax2.cla()
    ax2.plot(D)
    ax3 = fig.add_subplot(3,1,3)
    ax3.cla()
    ax3.plot(Y)
    plt.show()

def execute():
    global D,U,Dp,Up
    generate_weight_matrix()
    D, U = generate_data_sequence2()
    Dp = fsgm(D)
    Up = fsgm(U)
    Ds, Us = generate_s_sequence()

    train_network()
    test_network()
    plot2()

if __name__ == "__main__":
    execute()
