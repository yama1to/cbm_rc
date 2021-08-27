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
from generate_data_sequence import *
from generate_data_sequence_speech import generate_coch
from generate_matrix import *

class Config():
    def __init__(self):
        # columns, csv, id: データの管理のために必須の変数
        self.columns = None # 結果をCSVに保存する際のコラム
        self.csv = None # 結果を保存するファイル
        self.id  = None
        self.plot = 1#False # 図の出力のオンオフ
        self.show = 1#False # 図の表示（plt.show()）のオンオフ、explorerは実行時にこれをオフにする。
        self.savefig = False
        self.fig1 = "fig1.png" ### 画像ファイル名

        # config
        self.dataset=7
        self.seed:int=1 # 乱数生成のためのシード
        self.NN=200 # １サイクルあたりの時間ステップ
        self.MM=250 # サイクル数
        self.MM0 = 0 #

        self.Nu = 86   #size of input
        self.Nh:int = 100 #size of dynamical reservior
        self.Ny = 10   #size of output

        self.Temp=1
        self.dt=1.0/self.NN #0.01

        #sigma_np = -5
        self.alpha_i = 1.498
        self.alpha_r = 0.892
        self.alpha_b = 0.
        self.alpha_s = 1.998

        self.alpha0 = 0#0.1
        self.alpha1 = 0#-5.8

        self.beta_i = 0.1
        self.beta_r = 0.1
        self.beta_b = 0.1

        self.lambda0 = 0.011

        # Results
        self.RMSE1=None
        self.RMSE2=None
        self.cnt_overflow=None
        self.WER = None

def generate_weight_matrix():
    global Wr, Wb, Wo, Wi
    Wr = generate_random_matrix(c.Nh,c.Nh,c.alpha_r,c.beta_r,distribution="one",normalization="sr")
    Wb = generate_random_matrix(c.Nh,c.Ny,c.alpha_b,c.beta_b,distribution="one",normalization="none")
    Wi = generate_random_matrix(c.Nh,c.Nu,c.alpha_i,c.beta_i,distribution="one",normalization="none")
    Wo = np.zeros(c.Nh * c.Ny).reshape((c.Ny, c.Nh))

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
    #hx = np.zeros(Nh)
    hx = np.random.uniform(0,1,c.Nh) # [0,1]の連続値
    hs = np.zeros(c.Nh) # {0,1}の２値
    hs_prev = np.zeros(c.Nh)
    hc = np.zeros(c.Nh) # ref.clockに対する位相差を求めるためのカウント
    hp = np.zeros(c.Nh) # [-1,1]の連続値
    ht = np.zeros(c.Nh) # {0,1}

    Yp = np.zeros((c.MM, c.Ny))
    Yx = np.zeros((c.MM*c.NN, c.Ny))
    Ys = np.zeros((c.MM*c.NN, c.Ny))
    #ysign = np.zeros(Ny)
    yp = np.zeros(c.Ny)
    yx = np.zeros(c.Ny)
    ys = np.zeros(c.Ny)
    #yc = np.zeros(Ny)

    Us = np.zeros((c.MM*c.NN, c.Nu))
    Ds = np.zeros((c.MM*c.NN, c.Ny))
    Rs = np.zeros((c.MM*c.NN, 1))
    m = 0
    rs = 0
#  
    for n in range(c.NN * c.MM):
        theta = np.mod(n/c.NN,1) # (0,1)
        rs_prev = rs
        hs_prev = hs.copy()

        rs = p2s(theta,0)       # 参照クロック
        #print(Up.shape,m,n,c.NN*c.MM)
        us = p2s(theta,Up[m])    # エンコードされた入力
        #ds = p2s(theta,Dp[m])   #
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
            print(Wo.shape,hp.shape)#(10,100),(100,)
            print(yp.shape)#(10,)
            print(m)
            # record
            Hp[m]=hp
            Yp[m]=yp
            m+=1

        any_hs_change = np.any(hs!=hs_prev)

        # record
        Rs[n]=rs
        Hx[n]=hx
        Hs[n]=hs
        Yx[n]=yx
        Ys[n]=ys
        Us[n]=us
        #Ds[n]=ds

    # オーバーフローを検出する。
    global cnt_overflow
    cnt_overflow = 0
    for m in range(2,c.MM-1):
        tmp = np.sum( np.heaviside( np.fabs(Hp[m+1]-Hp[m]) - 0.6 ,0))
        cnt_overflow += tmp
        #print(tmp)

def train_network():
    global Wo,Up,Dp,Hp
    num,len,dim = U1.shape

    collecting_reservoir_state = np.empty((0,c.Nh))
    for i in range(num):
        print("train-dataset:",i)
        Up = U1[i]
        c.MM = len-1
        run_network(1) # run netwrok with teacher forcing
        collecting_reservoir_state = np.vstack((collecting_reservoir_state,Hp))
    

    #print(collecting_reservoir_state)   
    #print(collecting_reservoir_state.shape)     #48500,100 = dateset*len,Nh

    #データセット一つ＝195時間長を１つにする。頻


    Hp = collecting_reservoir_state
    
    M = Hp[c.MM0:, :]                           #48500,100 = dateset*len,Nh
    #invD = fyi(Dp)
    invD = fyi(Dp)
    G = invD[c.MM0:, :]                         #num*len,c.Nh = (48500, 100)

    #print(invD.shape)                           #250,10 = dataset,digit

    #print("Hp\n",Hp)
    #print("M\n",M)

    ### Ridge regression

    
    E = np.identity(c.Nh)
    TMP1 = np.linalg.inv(M.T@M + c.lambda0 * E)
    #print(TMP1.shape)   #100,100
    #print(M.T.shape,G.shape)
    WoT = TMP1@M.T@G    #(100,100)@(100,48500)@(48500,100)
    #print(WoT.shape)
    Wo = WoT.T
    #print(Wo)
    #print("WoT\n", WoT)
    global y
    y = Wo@ collecting_reservoir_state.T

def test_network():
    global Wo,Up,Dp,Hp
    num,len,dim = U1.shape

    for i in range(num):
        print("test-dataset:",i)
        Up = U1[i]
        c.MM = len-1
        run_network(0)
        #collecting_reservoir_state = np.vstack((collecting_reservoir_state,Hp))

    

def plot1():
    fig=plt.figure(figsize=(20, 12))
    Nr=6
    ax = fig.add_subplot(Nr,1,1)
    ax.cla()
    ax.set_title("Up")
    ax.plot(Up)

    ax = fig.add_subplot(Nr,1,2)
    ax.cla()
    ax.set_title("Us")
    ax.plot(Us)
    ax.plot(Rs,"r:")
    #ax.plot(R2s,"b:")

    ax = fig.add_subplot(Nr,1,3)
    ax.cla()
    ax.set_title("Hx")
    ax.plot(Hx)

    ax = fig.add_subplot(Nr,1,4)
    ax.cla()
    ax.set_title("Hp")
    ax.plot(Hp)

    ax = fig.add_subplot(Nr,1,5)
    ax.cla()
    ax.set_title("Yp")
    ax.plot(train_Y)

    ax = fig.add_subplot(Nr,1,6)
    ax.cla()
    ax.set_title("Dp")
    ax.plot(Dp)
    ax.plot()
    
    plt.show()
    plt.savefig(c.fig1)

def execute(c):
    global D,Ds,Dp,U,Us,Up,Rs,R2s,MM,D1,U1
    global RMSE1,RMSE2
    global train_Y

    np.random.seed(c.seed)

    generate_weight_matrix()

    ### generate data
    #  時系列データをコクリアグラムに変換し入力とする
    #  コクリアグラムがなんの数字に対応しているか one-hotベクトルを作成しTARGETとする
    #

    if c.dataset==7:

        train, valid, train_target, valid_target = generate_coch()
        U1 = train
        U2 = valid
        D1 = train_target
        D2 = valid_target

    print(U1.shape, D1.shape,   U2.shape,   D2.shape)
    #(250, 195, 86) (250*195, 10) (250, 195, 86) (250*195, 10)
    #データセット数、時間長、周波数チャネル

    ### training
    #print("training...")

    Dp = fy(D1)                      #one-hot vector
    Up = fy(U1)
    print(Dp.shape,Up.shape)
    train_network()                     #Up,Dpからネットワークを学習する

    Y_pred = fyi(Yp)
    print("yp",Y_pred.shape)
    Y_pred = y 
    print("yp",Y_pred.shape)
    test_length = 195

    pred_test = np.empty((0, 10))
    start = 0

    #195=１つのデータをまとめる
    for i in range(250):
        print(i)
        tmp = Y_pred[:,start:start+test_length]  # 1つのデータに対する出力
        max_index = np.argmax(tmp, axis=0)  # 最大出力を与える出力ノード番号
        histogram = np.bincount(max_index)  # 出力ノード番号のヒストグラム
        print(histogram.shape)
        pred_test = np.hstack((pred_test, np.argmax(histogram)))  # 最頻値
        start = start + test_length
    print(pred_test.shape)
    
    Dp = np.zeros((250,10))
    for i in range(10):
        Dp[25*i:25*(i+1)][-i-1] = 1
    count = np.sum(pred_test,Dp)
    print(count)
        




    ### test
    #print("test...")
    Up = fy(U2)
    Dp = fy(D2)
    test_network()                      #output = Yp

    Y_pred = fyi(Yp)
    print(Y_pred.shape)


    

    
    
    # 評価　Word Error Rate
    train_Y = fyi(Yp)       # PRED   one-hot vector (one-hot vec,dim)
    Dp = D2                 # TARGET one-hot vector

    WER = np.sum(train_Y - Dp) / 250
    print(WER)



    ######################################################################################
     # Results
    c.RMSE1=None
    c.RMSE2=None
    c.cnt_overflow=cnt_overflow
    c.WER = WER
    #####################################################################################

    if c.plot: plot1()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-config", type=str)
    a = ap.parse_args()

    c=Config()
    if a.config: c=common.load_config(a)
    execute(c)
    if a.config: common.save_config(c)