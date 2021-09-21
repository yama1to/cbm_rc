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
from generate_data_sequence_speech4 import *
from generate_matrix import *
from tqdm import tqdm

class Config():
    def __init__(self):
        # columns, csv, id: データの管理のために必須の変数
        self.columns = None # 結果をCSVに保存する際のコラム
        self.csv = None # 結果を保存するファイル
        self.id  = None
        self.plot = True # 図の出力のオンオフ
        self.show = True # 図の表示（plt.show()）のオンオフ、explorerは実行時にこれをオフにする。
        self.savefig = True
        self.fig1 = "fig1.png" ### 画像ファイル名
        self.tqdm_set = 1

        # config
        self.dataset=1
        self.seed:int=0 # 乱数生成のためのシード
        self.NN=256 # １サイクルあたりの時間ステップ
        self.MM=50 # サイクル数
        self.MM0 = 0 #

        self.Nu = 77   #size of input
        self.Nh = 300 #size of dynamical reservior
        self.Ny = 10   #size of output

        self.Temp=1.0
        self.dt=1.0/self.NN #0.01

        #sigma_np = -5
        self.alpha_i = 1
        self.alpha_r = 0.9
        self.alpha_b = 0.
        self.alpha_s = 2

        self.beta_i = 0.5
        self.beta_r = 0.1
        self.beta_b = 0.0

        self.lambda0 = 0.1

        # Results
        self.WER = None
        self.cnt_overflow=None

def generate_weight_matrix():
    global Wr, Wb, Wo, Wi
    Wr = generate_random_matrix(c.Nh,c.Nh,c.alpha_r,c.beta_r,distribution="one",normalization="sr")
    Wb = generate_random_matrix(c.Nh,c.Ny,c.alpha_b,c.beta_b,distribution="one",normalization="none")
    Wi = generate_random_matrix(c.Nh,c.Nu,c.alpha_i,c.beta_i,distribution="one",normalization="none")
    Wo = np.zeros(c.Nh * c.Ny).reshape(c.Ny, c.Nh)

def fy(h):
    return np.tanh(h)

def fyi(h):
    return np.arctanh(h)

def p2s(theta,p):
    return np.heaviside( np.sin(np.pi*(2*theta-p)),1)

def run_network(mode):
    global Hx, Hs, Hp, Y, Yx, Ys, Yp, Y, Us, Ds,Rs
    Hp = np.zeros((c.MM, c.Nh))
    Hx = np.zeros((c.MM*c.NN, c.Nh))
    Hs = np.zeros((c.MM*c.NN, c.Nh))
    hsign = np.zeros(c.Nh)
    hx = np.zeros(c.Nh)
    #hx = np.random.uniform(0,1,c.Nh) # [0,1]の連続値
    hs = np.zeros(c.Nh) # {0,1}の２値
    hs_prev = np.zeros(c.Nh)
    hc = np.zeros(c.Nh) # ref.clockに対する位相差を求めるためのカウント
    hp = np.zeros(c.Nh) # [-1,1]の連続値
    ht = np.zeros(c.Nh) # {0,1}

    Yp = np.zeros((c.MM, c.Ny))
    Yx = np.zeros((c.MM*c.NN, c.Ny))
    Ys = np.zeros((c.MM*c.NN, c.Ny))

    yp = np.zeros(c.Ny)
    yx = np.zeros(c.Ny)
    ys = np.zeros(c.Ny)

    Us = np.zeros((c.MM*c.NN, c.Nu))
    Ds = np.zeros((c.MM*c.NN, c.Ny))
    Rs = np.zeros((c.MM*c.NN, 1))

    rs = 1
    rs_prev = 0

    m=0
    for n in range(c.NN * c.MM):
        theta = np.mod(n/c.NN,1) # (0,1)
        rs_prev = rs
        hs_prev = hs.copy()

        rs = p2s(theta,0)# 参照クロック
        us = p2s(theta,Up[m]) # エンコードされた入力
        ds = p2s(theta,Dp[m]) #
        ys = p2s(theta,yp)

        sum = np.zeros(c.Nh)
        sum += c.alpha_s*rs # ラッチ動作を用いないref.clockと同期させるための結合
        #sum += alpha_s*(hs-rs)*ht # ref.clockと同期させるための結合
        sum += Wi@(2*us-1) # 外部入力
        sum += Wr@(2*hs-1) # リカレント結合

        #if mode == 0:
        #    sum += Wb@ys
        #if mode == 1:  # teacher forcing
        #    sum += Wb@ds

        hsign = 1 - 2*hs
        hx = hx + hsign*(1.0+np.exp(hsign*sum/c.Temp))*c.dt
        hs = np.heaviside(hx+hs-1,0)
        hx = np.fmin(np.fmax(hx,0),1)

        if rs==1:
            hc+=hs # デコードのためのカウンタ、ref.clockとhsのANDでカウントアップ

        # ref.clockの立ち上がり
        if rs_prev==0 and rs==1:
            hp = 2*hc/c.NN-1 # デコード、カウンタの値を連続値に変換
            hc = np.zeros(c.Nh) #カウンタをリセット
            #ht = 2*hs-1 リファレンスクロック同期用ラッチ動作をコメントアウト
            yp = fy(Wo@hp)
            # record
            Hp[m]=hp
            Yp[m]=yp
            m+=1

        any_hs_change = np.any(hs!=hs_prev)
        if c.plot:
            # record
            Rs[n]=rs
            Hx[n]=hx
            Hs[n]=hs
            Yx[n]=yx
            Ys[n]=ys
            Us[n]=us
            Ds[n]=ds

    # オーバーフローを検出する。
    global cnt_overflow
    cnt_overflow = 0
    for m in range(2,c.MM-1):
        tmp = np.sum( np.heaviside( np.fabs(Hp[m+1]-Hp[m]) - 0.6 ,0))
        cnt_overflow += tmp

def train_network():
    global Wo

    run_network(1) # run netwrok with teacher forcing

def test_network():
    run_network(0)

def plot1():
    fig=plt.figure(figsize=(20, 12))
    Nr=3
    ax = fig.add_subplot(Nr,1,1)
    ax.cla()
    ax.set_title("Up")
    ax.plot(UP)

    ax = fig.add_subplot(Nr,1,2)
    ax.cla()
    ax.set_title("Hp")
    ax.plot(collect_state_matrix)

    ax = fig.add_subplot(Nr,1,3)
    ax.cla()
    ax.set_title("Yp,dp")
    ax.plot(pred_test)
    ax.plot(Dp)


    plt.show()
    plt.savefig(c.fig1)

def execute():
    global D,Ds,Dp,U,Us,Up,Rs,R2s,MM
    global collect_state_matrix,target_matrix
    global UP,DP,pred_test,dp,test_WER
    c.Nh = int(c.Nh)
    if c.seed>=0:
        np.random.seed(int(c.seed))

    generate_weight_matrix()

    ### generate data
    if c.dataset==1:
        U1,U2,D1,D2, SHAPE = load_datasets()

    (dataset_num,length,c.Nu) = SHAPE

    U1 = U1[:dataset_num*length]
    U2 = U2[:dataset_num*length]
    D1 = D1[:dataset_num*length]
    D2 = D2[:dataset_num*length]

  
    

    ### training ######################################################################
    #print("training...")

    DP = D1                     #one-hot vector
    UP = U1
    
    x = U1.shape[0]
    collect_state_matrix = np.empty((x,c.Nh))
    
    start = 0
    target_matrix = DP.copy()
    #print(np.linalg.pinv(collect_state_matrix).T.shape,target_matrix.T.shape)

    if c.tqdm_set:
        bar = tqdm(total = dataset_num) 
        bar.set_description("training...")

    for i in range(dataset_num):
        if c.tqdm_set:
            bar.update(1)

        Dp = DP[start:start + length]
        Up = UP[start:start + length]
        
        train_network()                     #Up,Dpからネットワークを学習する
 
        collect_state_matrix[start:start + length,:] = Hp
        
        start += length
        
        

    #weight matrix
    #"""
    #ridge reg
    M = collect_state_matrix[c.MM0:]
    G = target_matrix[c.MM0:]

    
    Wout = np.dot(G.T,np.linalg.pinv(M).T)
    Y_pred = Wout @ M.T

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

    #print("train Word error rate:",train_WER)

        
    ### test ######################################################################
    #print("test...")
    DP = D2                 #one-hot vector
    UP = U2

    start = 0

    target_matrix = Dp.copy()

    if c.tqdm_set:
        bar = tqdm(total = dataset_num) 
        bar.set_description("test...")

    for i in range(c.MM0,dataset_num):
        if c.tqdm_set:
            bar.update(1)

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
    
    dp = [DP[i] for i in range(0,U1.shape[0],length)]
    dp = dp
    test_WER = np.sum(abs(pred_test-dp)/2)/dataset_num
    print("train vs test :",train_WER,test_WER)


    ################################################################################
    c.WER = test_WER
    c.cnt_overflow = cnt_overflow
    ################################################################################

    if c.plot: plot1()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-config", type=str)
    a = ap.parse_args()

    c=Config()
    if a.config: c=common.load_config(a)
    execute()
    if a.config: common.save_config(c)
