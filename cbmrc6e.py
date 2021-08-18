# Copyright (c) 2018-2021 Katori lab. All Rights Reserved
"""
NOTE: cbm_rc　時系列生成 explorer2に対応

"""
import argparse
import numpy as np
import scipy.linalg
from numpy.linalg import svd, inv, pinv

import matplotlib as mpl
#mpl.use('Agg')## サーバ上で画像を出力するための設定。ローカルで表示する際はコメントアウトする
import matplotlib.pyplot as plt

import sys
import copy
from explorer import common
from generate_data_sequence import *

class Config():#Configクラスによりテストとメインの間で設定と結果をやりとりする。
    def __init__(self):
        # columns, csv, id: データの管理のために必須の変数
        self.columns = None # 結果をCSVに保存する際のコラム
        self.csv = None # 結果を保存するファイル
        self.id  = None
        self.plot = True # 図の出力のオンオフ
        self.show = True # 図の表示（plt.show()）のオンオフ、explorerは実行時にこれをオフにする。
        self.savefig = True
        # NOTE: optimization, gridsearch, randomsearchは、実行時にplot,show,savefig属性をFalseに設定する。
        self.fig1 = "data_cbmrc6e_fig1.png" ### 画像ファイル名

        ### config
        self.dataset=1
        self.seed=10 # 乱数生成のためのシード
        self.NN=200
        self.MM=300
        self.MM0 = 50

        self.Nu = 2   #size of input
        self.Nh = 100 #size of dynamical reservior
        self.Ny = 2   #size of output

        self.Temp=1
        self.dt=1.0/self.NN #0.01

        #sigma_np = -5
        self.alpha_i = 0.2
        self.alpha_r = 0.25
        self.alpha_b = 0.
        self.alpha_s = 0.6

        self.alpha0 = 0#0.1
        self.alpha1 = 0#-5.8

        self.beta_i = 0.1
        self.beta_r = 0.1
        self.beta_b = 0.1

        #tau = 2
        self.lambda0 = 0.1


def config(c):
    global dataset,seed
    global NN,MM,MM0,Nu,Nh,Ny,Temp,dt
    global alpha_i,alpha_r,alpha_b,alpha_s,alpha0,alpha1
    global beta_i,beta_r,beta_b
    global lambda0

    dataset = c.dataset
    seed = int(c.seed) # 乱数生成のためのシード
    NN = int(c.NN)
    MM = int(c.MM)
    MM0 = int(c.MM0)

    Nu = int(c.Nu)   #size of input
    Nh = int(c.Nh) #size of dynamical reservior
    Ny = int(c.Ny)   #size of output

    Temp=c.Temp
    dt=1.0/c.NN #0.01

    #sigma_np = -5
    alpha_i = c.alpha_i
    alpha_r = c.alpha_r
    alpha_b = c.alpha_b
    alpha_s = c.alpha_s

    alpha0 = c.alpha0
    alpha1 = c.alpha1

    beta_i = c.beta_i
    beta_r = c.beta_r
    beta_b = c.beta_b

    #tau = 2
    lambda0 = c.lambda0

def generate_random_matrix(n,m,beta):
    W = np.zeros(n*m)
    nonzeros = n * m * beta
    W[0:int(nonzeros / 2)] = 1
    W[int(nonzeros / 2):int(nonzeros)] = -1
    np.random.shuffle(W)
    W = W.reshape((n,m))
    return W

def generate_weight_matrix():
    global Wr, Wb, Wo, Wi
    ### Wr
    Wr0=generate_random_matrix(Nh,Nh,beta_r)
    v = scipy.linalg.eigvals(Wr0)
    lambda_max = max(abs(v))
    #print("WoT\n", WoT)
    Wr = Wr0 / lambda_max * alpha_r
    E = np.identity(Nh)
    Wr = Wr + alpha0*E
    #Wr = Wr + alpha1
    Wr = Wr + alpha1/Nh

    ### Wb
    Wb = generate_random_matrix(Nh,Ny,beta_b)
    Wb = Wb * alpha_b

    ### Wi
    Wi = generate_random_matrix(Nh,Nu,beta_i)
    Wi = Wi * alpha_i

    ### Wo
    Wo = np.zeros(Nh * Ny).reshape((Ny, Nh))
    # print(Wo)

def fx(h):
    return np.tanh(h)

def fy(h):
    return np.tanh(h)

def fyi(h):
    #print("WoT\n", WoT)
    return np.arctanh(h)
    #return -np.log(1.0/h-1.0)
def fr(h):
    return np.fmax(0, h)

def fsgm(h):
    return 1.0/(1.0+np.exp(-h))

def p2s(theta,p):
    return np.heaviside( np.sin(np.pi*(2*theta-p)),1)

def run_network(mode):
    global Hx, Hs, Hp, Y, Yx, Ys, Yp, Y, Us, Ds,Rs
    Hp = np.zeros((MM, Nh))
    Hx = np.zeros((MM*NN, Nh))
    Hs = np.zeros((MM*NN, Nh))
    hsign = np.zeros(Nh)
    #hx = np.zeros(Nh)
    hx = np.random.uniform(0,1,Nh) # [0,1]の連続値
    hs = np.zeros(Nh) # {0,1}の２値
    hs_prev = np.zeros(Nh)
    hc = np.zeros(Nh) # ref.clockに対する位相差を求めるためのカウント
    hp = np.zeros(Nh) # [-1,1]の連続値
    ht = np.zeros(Nh) # {0,1}

    Yp = np.zeros((MM, Ny))
    Yx = np.zeros((MM*NN, Ny))
    Ys = np.zeros((MM*NN, Ny))
    #ysign = np.zeros(Ny)
    yp = np.zeros(Ny)
    yx = np.zeros(Ny)
    ys = np.zeros(Ny)
    #yc = np.zeros(Ny)

    Us = np.zeros((MM*NN, Nu))
    Ds = np.zeros((MM*NN, Ny))
    Rs = np.zeros((MM*NN, 1))

    rs = 1
    rs_prev = 0
    count=0
    m=0
    for n in range(NN*MM):
        theta = np.mod(n/NN,1) # (0,1)
        rs_prev = rs
        hs_prev = hs.copy()

        rs = p2s(theta,0)
        us = p2s(theta,Up[m])
        ds = p2s(theta,Dp[m])
        ys = p2s(theta,yp)

        sum = np.zeros(Nh)
        sum += alpha_s*rs # ラッチ動作を用いないref.clockと同期させるための結合
        #sum += alpha_s*(hs-rs)*ht # ref.clockと同期させるための結合
        sum += Wi@(2*us-1) # 外部入力
        sum += Wr@(2*hs-1) # リカレント結合

        #if mode == 0:
        #    sum += Wb@ys
        #if mode == 1:  # teacher forcing
        #    sum += Wb@ds

        hsign = 1 - 2*hs
        hx = hx + hsign*(1.0+np.exp(hsign*sum/Temp))*dt
        hs = np.heaviside(hx+hs-1,0)
        hx = np.fmin(np.fmax(hx,0),1)

        if rs==1: hc+=hs # デコードのためのカウンタ、ref.clockとhsのANDでカウントアップ
        count = count + 1

        # ref.clockの立ち上がり
        if rs_prev==0 and rs==1:
            hp = 2*hc/NN-1
            hc = np.zeros(Nh) #カウンタをリセット
            #ht = 2*hs-1 リファレンスクロック同期用ラッチ動作をコメントアウト
            yp = np.tanh(Wo@hp)
            #yp=fsgm(Wo@hp)
            count=0
            # record
            Hp[m]=hp
            Yp[m]=yp
            m+=1

        # record
        Rs[n]=rs
        Hx[n]=hx
        Hs[n]=hs
        Yx[n]=yx
        Ys[n]=ys
        Us[n]=us
        Ds[n]=ds

    # 不連続な値の変化を検出する。
    global count_gap
    count_gap = 0
    for m in range(2,MM-1):
        tmp = np.sum( np.heaviside( np.fabs(Hp[m+1]-Hp[m]) - 0.6 ,0))
        count_gap += tmp
        #print(tmp)

def train_network():
    global Wo

    run_network(1) # run netwrok with teacher forcing

    M = Hp[MM0:, :]
    invD = fyi(Dp)
    G = invD[MM0:, :]

    #print("Hp\n",Hp)
    #print("M\n",M)

    ### Ridge regression
    E = np.identity(Nh)
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
    ax.plot(Yp)

    ax = fig.add_subplot(Nr,1,6)
    ax.cla()
    ax.set_title("Dp")
    ax.plot(Dp)

    plt.show()
    plt.savefig(c.fig1)

def execute():
    global D,Ds,Dp,U,Us,Up,Rs,R2s,MM
    global RMSE1,RMSE2
    
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
        sum += (Yp[j] - Dp[j])**2
    SUM=np.sum(sum)
    RMSE1 = np.sqrt(SUM/Ny/(MM-MM0))
    RMSE2 = 0
    #print(RMSE1)
    c.RMSE1=RMSE1
    c.RMSE2=RMSE2

    if c.plot :plot1()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-config", type=str)
    a = ap.parse_args()

    c=Config()
    if a.config: c=common.load_config(a)
    config(c)
    execute()
    if a.config: common.save_config(c)
