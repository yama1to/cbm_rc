# Copyright (c) 2018-2021 Katori lab. All Rights Reserved
"""
NOTE: cbm_rc　時系列生成タスク　
cbmrc6e.pyを改変
Configクラスによるパラメータ設定
"""

import argparse
import numpy as np
from numpy.linalg.linalg import norm
import scipy.linalg
import matplotlib.pyplot as plt
import copy
import time
from explorer import common
from generate_data_sequence_santafe import *
from generate_matrix import *
import gc

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

        # config
        self.dataset=1
        self.seed:int=0 # 乱数生成のためのシード
        self.NN=256 # １サイクルあたりの時間ステップ
        self.MM=300 # サイクル数
        self.MM0 = 0 #

        self.Nu = 1   #size of input
        self.Nh = 100 #size of dynamical reservior
        self.Ny = 5   #size of output

        self.Temp=1.0
        self.dt=1.0/self.NN #0.01

        #sigma_np = -5
        self.alpha_i = 0.25
        self.alpha_r = 0.36
        self.alpha_b = 0.
        self.alpha_s = 0.82

        self.beta_i = 0.4
        self.beta_r = 0.78
        self.beta_b = 0.1

        self.lambda0 = 0.1
        self.future = 1

        # Results
        self.RMSE=None
        self.NRMSE=None
        self.NRMSE2 =None
        self.NMSE2 =None
        self.NMSE = None
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
    #ysign = np.zeros(Ny)
    yp = np.zeros(c.Ny)
    yx = np.zeros(c.Ny)
    ys = np.zeros(c.Ny)
    #yc = np.zeros(Ny)

    Us = np.zeros((c.MM*c.NN, c.Nu))
    Ds = np.zeros((c.MM*c.NN, c.Ny))
    Rs = np.zeros((c.MM*c.NN, 1))

    rs = 1
    any_hs_change = True
    count =0
    m = 0
    for n in tqdm(range(c.NN * c.MM)):
        theta = np.mod(n/c.NN,1) # (0,1)
        rs_prev = rs
        hs_prev = hs.copy()

        rs = p2s(theta,0)# 参照クロック
        us = p2s(theta,Up[m]) # エンコードされた入力
        ds = p2s(theta,Dp[m]) #
        ys = p2s(theta,yp)

        sum = np.zeros(c.Nh)
        #sum += c.alpha_s*rs # ラッチ動作を用いないref.clockと同期させるための結合
        sum += c.alpha_s*(hs-rs)*ht # ref.clockと同期させるための結合
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

        hc[(hs_prev == 1)& (hs==0)] = count
        
        # ref.clockの立ち上がり
        if rs_prev==0 and rs==1:
            hp = 2*hc/c.NN-1 # デコード、カウンタの値を連続値に変換
            hc = np.zeros(c.Nh) #カウンタをリセット
            ht = 2*hs-1 #リファレンスクロック同期用ラッチ動作をコメントアウト
            yp = Wo@hp
            # record    
            Hp[m]=hp
            Yp[m]=yp
            count = 0
            m += 1

        #境界条件
        if n == (c.NN * c.MM-1):
            hp = 2*hc/c.NN-1 # デコード、カウンタの値を連続値に変換
            yp = Wo@hp
            # record
            Hp[m]=hp
            Yp[m]=yp

        count += 1
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
        #print(tmp)

def train_network():
    global Wo

    run_network(1) # run netwrok with teacher forcing

    M = Hp[c.MM0:, :]
    invD = Dp
    G = invD[c.MM0:, :]

    #print("Hp\n",Hp)
    #print("M\n",M)

    if c.lambda0 == 0:
        Wo = np.dot(G.T,np.linalg.pinv(M).T)
        #print("a")
    else:
        E = np.identity(c.Nh)
        TMP1 = np.linalg.inv(M.T@M + c.lambda0 * E)
        WoT = TMP1@M.T@G
        Wo = WoT.T
    #print("WoT\n", WoT)

def test_network():
    run_network(0)
def plot1():
    fig=plt.figure(figsize=(20, 12))
    Nr=4
    ax = fig.add_subplot(Nr,1,1)
    ax.cla()
    #ax.set_title("input")
    ax.plot(Up)

    ax = fig.add_subplot(Nr,1,2)
    ax.cla()
    #ax.set_title("decoded reservoir states")
    ax.plot(Hp)

    ax = fig.add_subplot(Nr,1,3)
    ax.cla()
    #ax.set_title("predictive output")
    #ax.plot(train_Y)
    ax.plot(Yp)

    ax = fig.add_subplot(Nr,1,4)
    ax.cla()
    #ax.set_title("desired output")
    ax.plot(Dp)
    plt.savefig("./eps-fig/santafe.eps")

def plot2():
    fig=plt.figure(figsize=(20, 12))
    Nr=2
    ax = fig.add_subplot(Nr,1,1)
    ax.cla()
    l = Dp.shape[1]
    for i in range(1,l+1):
        ax.plot(list(range(train_num,train_num+test_num)),Yp[:,i-1], label = "prediction:future={}".format(i))
        ax.plot(list(range(train_num,train_num+test_num)),Dp[:,i-1], label = "Target:future={}".format(i))
    ax.legend()
    # ax.set_xlabel("time")
    # ax.set_ylabel("value of prediction and target")

    ax = fig.add_subplot(Nr,1,2)
    ax.cla()

    error = abs(Yp-Dp)
    for i in range(1,l+1):
        ax.plot(list(range(train_num,train_num+test_num)),error[:,i-1],label ="error:future={}".format(i) )
    # ax.set_xlabel("time")
    # ax.set_ylabel("absolute value of error")
    plt.legend()
    plt.savefig("santafe-error.eps")
    plt.tight_layout()
    #plt.show()

def execute(c):
    global D,Ds,Dp,U,Us,Up,Rs,R2s,MM
    global RMSE1,RMSE2,Yp,normalize,train_num,test_num,future
    t_start=time.time()
    #if c.seed>=0:
    np.random.seed(int(c.seed))
    #np.random.seed(c.seed)

    
    c.Nh = int(c.Nh)
    ### generate data
    train_num = 1000
    test_num = 2000
    if c.dataset==1:
        #future = 1,2,3,4,5
        c.future = [1,2,3,4,5]
        U1,D1,U2,D2,normalize = generate_santafe(future = c.future,train_num = train_num,test_num =test_num,)
        # plt.plot(list(range(1000)),U1)
        # plt.plot(list(range(1000,3000)),U2)
        # plt.tight_layout()
        # #plt.show()
        # plt.savefig("santafe-u.eps")
    #print(D2[:,2]==D2[:,3])
    # plt.plot(U2,label="u")
    # for i in range(5):
    #     plt.plot(D2[:,i],label=i+1)
    # plt.legend()
    # plt.show()
    #print(normalize)
    c.Ny = int(len(c.future))
    generate_weight_matrix()
    ### training
    #print("training...")
    c.MM= U1.size

    Dp = D1
    Up = U1
    if c.plot:
        del U1,D1
        gc.collect()

    train_network()

    


    ### test
    #print("test...")
    c.MM= U2.size

    Dp = D2
    Up = U2
    if c.plot:
        del U2,D2
        gc.collect()
    test_network()


    ### evaluation
    sum=0
    Yp = Yp*normalize
    Dp = Dp*normalize
 
    for j in range(c.MM):
        sum += (Yp[j] - Dp[j])**2
    MSE = sum/c.MM
    RMSE = np.sqrt(MSE)
    
    NRMSE = RMSE/np.var(Dp)#np.std(Dp)#np.var(Dp)
    c.RMSE = RMSE
    c.NRMSE2 = NRMSE

    NMSE =(MSE/np.var(Dp))
    print(NMSE)
    c.NMSE = np.sum(NMSE)/NMSE.size
    c.NRMSE = np.sum(NRMSE)/NRMSE.size
    c.cnt_overflow=cnt_overflow
    c.NMSE2 = NMSE
    #print(RRMSE1)
    #print("それぞれのfutureでのNRMSEを全部加算した場合のNRMSE: "+str(c.NRMSE))
    #print("time: %.6f [sec]" % (time.time()-t_start))

    if c.plot: 
        #plot1()
        plot2()
        #print(RMSE)
        print("それぞれのfutureでのNMSEを全部加算した場合のNMSE: "+str(c.NMSE))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-config", type=str)
    a = ap.parse_args()

    c=Config()
    if a.config: c=common.load_config(a)
    execute(c)
    if a.config: common.save_config(c)
