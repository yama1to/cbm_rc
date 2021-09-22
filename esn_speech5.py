# Copyright (c) 2018-2021 Katori lab. All Rights Reserved
"""
NOTE: cbm_rc　時系列生成タスク　
cbmrc6e.pyを改変
Configクラスによるパラメータ設定
"""

import argparse
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import copy
import time
from explorer import common
from generate_data_sequence_speech5 import *
from generate_matrix import *
from tqdm import tqdm
from pprint import pprint
import gc

class Config():
    def __init__(self):
        # columns, csv, id: データの管理のために必須の変数
        self.columns = None # 結果をCSVに保存する際のコラム
        self.csv = None # 結果を保存するファイル
        self.id  = None
        self.plot = True # 図の出力のオンオフ
        self.show = False # 図の表示（plt.show()）のオンオフ、explorerは実行時にこれをオフにする。
        self.savefig = False
        self.fig1 = "fig1.png" ### 画像ファイル名
        self.isNotUseTqdm = True

        # config
        self.dataset=6
        self.seed:int=1 # 乱数生成のためのシード
        self.MM=50 # サイクル数
        self.MM0 = 0 #

        self.Nu = 77   #size of input
        self.Nh:int = 200#815 #size of dynamical reservior
        self.Ny = 10   #size of output


        #sigma_np = -5
        self.alpha_i = 907
        self.alpha_r = 0.7
        self.alpha_b = 0.

        self.alpha0 = 0.15#0.1
        self.alpha1 = 0#-5.8

        self.beta_i = 0.03
        self.beta_r = 0.1
        self.beta_b = 0.1

        self.lambda0 = 0.

        # Results
        self.train_WER = None
        self.WER=None



def generate_weight_matrix():
    global Wr, Wb, Wo, Wi
    Wr = generate_random_matrix(c.Nh,c.Nh,c.alpha_r,c.beta_r,distribution="one",normalization="sr")
    Wb = generate_random_matrix(c.Nh,c.Ny,c.alpha_b,c.beta_b,distribution="one",normalization="none")
    Wi = generate_random_matrix(c.Nh,c.Nu,c.alpha_i,c.beta_i,distribution="one",normalization="none")
    Wo = np.zeros(c.Nh * c.Ny).reshape((c.Ny, c.Nh))

def fy(h):
    return np.tanh(h)

def run_network(mode):
    global Hp
    
    Hp = np.zeros((c.MM, c.Nh))
    #x = np.random.uniform(-1, 1, Nh)/ 10**4
    x = np.zeros(c.Nh)
    

    for n in range(c.MM - 1):
        
        u = Up[n, :]

        #Hp[n+1,:] = x + 1.0/tau * (-alpha0 * x + fx(Wi@u + Wr@x))
        next_x = (1 - c.alpha0) * x + c.alpha0*fy(Wi@u + Wr@x)
        Hp[n+1,:] = next_x
        x= next_x

        
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


def execute(c):
    global D,Ds,Dp,U,Us,Up,Rs,R2s,MM,Yp
    global RMSE1,RMSE2
    global train_Y_binary,MC,DC
    global capacity ,MM,Dp,Up,Nh,Nx,X ,Nu
    global UP,DP,pred_test,dp,test_WER
    global Wi,Wo,Wr
    global collect_state_matrix,target_matrix,Y_pred,pred_test
    global U1,U2,D1,D2,SHAPE

    c.Nh = int(c.Nh)

    np.random.seed(seed = int(c.seed))
    generate_weight_matrix()

    ### generate data
    
    U1,U2,D1,D2,SHAPE = load_datasets()
    (dataset_num,length,Nu) = SHAPE


    ### training
    #print("training...")
    

    #Scale to (-1,1)
    DP = D1                     # TARGET   #(MM,len(delay))   
    UP = U1                     # INPUT    #(MM,1)
    
    x = U1.shape[0]
    collect_state_matrix = np.empty((x,c.Nh))
    start = 0
    target_matrix = DP.copy()
    
    del U1,D1
    gc.collect()

    for _ in tqdm(range(dataset_num),disable=c.isNotUseTqdm):
        Dp = DP[start:start + length]
        Up = UP[start:start + length]
        
        train_network()                     #Up,Dpからネットワークを学習する
 
        collect_state_matrix[start:start + length,:] = Hp
        
        start += length

    #weight matrix
    #"""
    #ridge reg
    M = collect_state_matrix[c.MM0:]
    G = target_matrix

    #Wout = np.linalg.inv(M.T@M + c.lambda0 * np.identity(c.Nh)) @ M.T @ G
    #print(G.shape,M.shape)
    Wout = np.dot(G.T,np.linalg.pinv(M).T)
    #print(Wout.shape,M.shape)
    Y_pred = Wout @ M.T
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

    dp = [DP[i] for i in range(0,x,length)]

    train_WER = np.sum(abs(pred_train-dp)/2)/dataset_num 

    #print("train Word error rate:",train_WER)

    
    ### test ######################################################################
    #print("test...")
    DP = D2                 #one-hot vector
    UP = U2
    del U2,D2
    gc.collect()
    start = 0

    target_matrix = Dp.copy()
    for i in tqdm(range(c.MM0,dataset_num),disable=c.isNotUseTqdm):
        Dp = DP[start:start + length]
        Up = UP[start:start + length]

        test_network()                     #Up,Dpからネットワークを学習する

        collect_state_matrix[start:start + length,:] = Hp
        start += length

    Y_pred = Wout @ collect_state_matrix[c.MM0:].T

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
    
    dp = [DP[i] for i in range(0,UP.shape[0],length)]
    
    test_WER = np.sum(pred_test!=dp)/2/dataset_num
    #print("test Word error rate:",test_WER)
    print("train vs test :",train_WER,test_WER)
    # for i in range(250):
    #     print(pred_test[i],dp[i])
    display=0
    if display :
        plot2()
   
    #print(MC,MC1,MC2,MC3,MC4)
######################################################################################
     # Results
    c.RMSE1=None
    c.RMSE2=None
    c.train_WER = train_WER
    c.WER = test_WER
    
    
    #print("MC =",c.MC)

#####################################################################################
    if c.plot:
        plot2()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-config", type=str)
    a = ap.parse_args()

    c=Config()
    if a.config: c=common.load_config(a)
    execute(c)
    if a.config: common.save_config(c)
