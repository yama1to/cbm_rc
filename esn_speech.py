# Copyright (c) 2017-2018 Katori Lab. All Rights Reserved
# NOTE:ESNによる時系列の生成, random spikeによる評価, capacityの評価
# Furuta (2018)にある性能評価指標のcapacityを実装した
from generate_data_sequence_speech import generate_coch
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
from generate_matrix import *
from tqdm import tqdm

file_csv = "data_esn1.csv"
file_fig1 = "data_esn1_fig1.png"
display = 1

dataset = 7
seed=10 # 乱数生成のためのシード
id=0

MM = 50
MM0 = 0
T0 = 0

Nu = 86   #size of input
Nx = 400 #size of dynamical reservior
Ny = 10   #size of output

sigma_np = -5
alpha_r = 0.9
alpha_b = 0.0 # 0.8
alpha_i = 10**4
beta_r = 0.05
beta_b = 0 
beta_i = 0.5
alpha0 = 1
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
    % (dataset,seed,id,Nx,alpha_i,alpha_r,alpha_b,alpha0,tau,beta_i,beta_r,beta_b,lambda0)

    f=open(file_csv,"a")
    f.write(str)
    f.close()

def generate_weight_matrix():
    global Wr, Wb, Wo, Wi
    Wr = generate_random_matrix(Nh,Nh,alpha_r,beta_r,distribution="one",normalization="sr")
    Wb = generate_random_matrix(Nh,Ny,alpha_b,beta_b,distribution="one",normalization="none")
    Wi = generate_random_matrix(Nh,Nu,alpha_i,beta_i,distribution="one",normalization="none")
    Wo = np.zeros(Nh * Ny).reshape(Ny, Nh)


def fx(x):
    return np.tanh(x)

def fy(x):
    return np.tanh(x)

def fyi(x):
    return np.arctanh(x)

def fr(x):
    return np.fmax(0, x)

def run_network(mode):
    global X
    X = np.zeros((MM, Nx))

    #x = np.random.uniform(-1, 1, Nx)/ 10**4
    x = np.zeros(Nx)
    for n in range(MM - 1):
        u = Up[n, :]

        #X[n+1,:] = x + 1.0/tau * (-alpha0 * x + fx(Wi@u + Wr@x))
        X[n+1,:] = (1 - alpha0) * x + alpha0*fx(Wi@u + Wr@x)
def train_network():
    global Wo

    run_network(1) # run netwrok with teacher forcing

def test_network():
    run_network(0)

def plot2():
    fig=plt.figure(figsize=(20, 12))
    Nr=3
    ax = fig.add_subplot(Nr,1,1)
    ax.cla()
    ax.set_title("U")
    ax.plot(UP)

    ax = fig.add_subplot(Nr,1,2)
    ax.cla()
    ax.set_title("X")
    ax.plot(collect_state_matrix)

    ax = fig.add_subplot(Nr,1,3)
    ax.cla()
    ax.set_title("Y, Ytarget")
    ax.plot(dp)
    ax.plot(pred_test)

    plt.show()
    #plt.savefig(file_fig1)
    plot3(Y_pred.T)

def plot3(tmp):
    fig=plt.figure(figsize=(20, 12))
    Nr=10
    for i in range(1,11):
        ax = fig.add_subplot(Nr,1,i)
        ax.cla()
        ax.set_title("y")
        ax.plot(tmp[:,i-1])
    plt.show()

def execute():
    #global D,Ds,Dp,U,Us,Up,Rs,R2s,MM
    global RMSE1,RMSE2
    global capacity ,MM,Dp,Up,Nh,Nx,X ,Nu
    global UP,DP,pred_test,dp,test_WER
    global Wi,Wo,Wr
    global collect_state_matrix,target_matrix,Y_pred,pred_test

    Nh = Nx
    

    if seed>=0:
        np.random.seed(seed)
    generate_weight_matrix()

    #入力 2dim
    #U (312 * 250, 78) 
    #D (312 * 250, 10)
    U1,U2,D1,D2,SHAPE = generate_coch(seed = seed,shuffle = 0)
    (dataset_num,length,Nu) = SHAPE 


    ### training ######################################################################
    print("training...")
    # U1 = U1.T
    # U2 = U2.T
    # D1 = D1.T
    # D2 = D2.T
    MAX1 = np.max(np.max(U1,axis = 1),axis=0)
    MAX2 = np.max(np.max(U2,axis = 1),axis=0)
    MAX = max(MAX1,MAX2)

    #U1 = U1* 10**4
    #U2 = U2 * 10**4
    U1 = U1 /MAX
    U2 = U2 /MAX

    DP = D1                      #one-hot vector

    UP = fy(U1)

    #DP = D1 
    #UP = U1
    x = U1.shape[0]
    collect_state_matrix = np.empty((x,Nh))
    start = 0
    target_matrix = DP.copy()
    
    for _ in tqdm(range(dataset_num)):
        Dp = DP[start:start + length]
        Up = UP[start:start + length]
        
        train_network()                     #Up,Dpからネットワークを学習する
 
        collect_state_matrix[start:start + length,:] = X
        
        start += length

    #weight matrix
    #"""
    #ridge reg
    M = collect_state_matrix[MM0:]
    G = target_matrix

    Wout = inv(M.T@M + lambda0 * np.identity(Nh)) @ M.T @ G
    #Wout = np.dot(G.T,np.linalg.pinv(M).T)
    Y_pred = Wout.T @ M.T
    #Y_pred = np.dot(Wout , M.T)
    #"""

    pred_train = np.zeros((dataset_num,10))
    start = 0

    for i in range(dataset_num):

        tmp = Y_pred[:,start:start+length]  # 1つのデータに対する出力
        max_index = np.argmax(tmp, axis=0)  # 最大出力を与える出力ノード番号

        histogram = np.bincount(max_index)  # 出力ノード番号のヒストグラム
        idx = np.argmax(histogram)
        pred_train[i][idx] = 1              # 最頻値
        start = start + length

    dp = [DP[i] for i in range(0,U1.shape[0],length)]
    dp = dp
    train_WER = np.sum(abs(pred_train-dp)/2)/dataset_num 

    print("train Word error rate:",train_WER)

        
    ### test ######################################################################
    print("test...")
    DP = D2                 #one-hot vector
    UP = fy(U2)
    #DP = D2 
    #UP = U2
    start = 0

    target_matrix = Dp.copy()
    for i in tqdm(range(MM0,dataset_num)):
        Dp = DP[start:start + length]
        Up = UP[start:start + length]

        test_network()                     #Up,Dpからネットワークを学習する

        collect_state_matrix[start:start + length,:] = X
        start += length

    Y_pred = Wout.T @ collect_state_matrix[MM0:].T

    pred_test = np.zeros((dataset_num,10))
    start = 0

    #圧縮
    for i in range(dataset_num):
        tmp = Y_pred[:,start:start+length]  # 1つのデータに対する出力

        max_index = np.argmax(tmp, axis=0)  # 最大出力を与える出力ノード番号

        histogram = np.bincount(max_index)  # 出力ノード番号のヒストグラム
        idx = np.argmax(histogram)
        pred_test[i][idx] = 1  # 最頻値

        start = start + length
    
    dp = [DP[i] for i in range(0,U1.shape[0],length)]
    dp = dp
    test_WER = np.sum(abs(pred_test-dp)/2)/dataset_num
    print("test Word error rate:",test_WER)
    print("train vs test :",train_WER,test_WER)
    # for i in range(250):
    #     print(pred_test[i],dp[i])
    display=0
    if display :
        plot2()

if __name__ == "__main__":
    #config()
    execute()
    #output()
