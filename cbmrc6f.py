# Copyright (c) 2018 Katori lab. All Rights Reserved
"""
NOTE: cbm_rc　時系列生成　explorer2に対応。
Configクラスのインスタンスをプログラムの各所で使用する。
"c."を各所につける手間があり、型変換が必要になる場合もある。
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


        self.fig1 = "fig1.png" ### 画像ファイル名
        # config

        self.dataset=1
        self.seed:int=0 # 乱数生成のためのシード
        self.NN=200
        self.MM=300
        self.MM0 = 50

        self.Nu = 2   #size of input
        self.Nh:int = 100 #size of dynamical reservior
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

        # Results
        self.RMSE1=None
        self.RMSE2=None
        self.cnt_overflow=None

def generate_weight_matrix():
    global Wr, Wb, Wo, Wi
    ### Wr
    Wr0 = np.zeros(int(c.Nh * c.Nh))
    nonzeros = c.Nh * c.Nh * c.beta_r
    Wr0[0:int(nonzeros / 2)] = 1
    Wr0[int(nonzeros / 2):int(nonzeros)] = -1
    np.random.shuffle(Wr0)
    Wr0 = Wr0.reshape(int(c.Nh), int(c.Nh))
    v = scipy.linalg.eigvals(Wr0)
    lambda_max = max(abs(v))
    #print("WoT\n", WoT)
    Wr = Wr0 / lambda_max * c.alpha_r
    E = np.identity(c.Nh)
    Wr = Wr + c.alpha0*E
    #Wr = Wr + alpha1

    Wr = Wr + c.alpha1/c.Nh
    #Wr = Wr -0.06#/Nh

    # print("lamda_max",lambda_max)
    # print("Wr:")
    # print(Wr)

    ### Wb
    Wb = np.zeros(c.Nh * c.Ny)
    Wb[0:int(c.Nh * c.Ny * c.beta_b / 2)] = 1
    Wb[int(c.Nh * c.Ny * c.beta_b / 2):int(c.Nh * c.Ny * c.beta_b)] = -1
    np.random.shuffle(Wb)
    Wb = Wb.reshape((c.Nh, c.Ny))
    Wb = Wb * c.alpha_b
    # print("Wb:")
    # print(Wb)

    ### Wi
    Wi = np.zeros(c.Nh * c.Nu)
    Wi[0:int(c.Nh * c.Nu * c.beta_i / 2)] = 1
    Wi[int(c.Nh * c.Nu * c.beta_i / 2):int(c.Nh * c.Nu * c.beta_i)] = -1
    np.random.shuffle(Wi)
    Wi = Wi.reshape((c.Nh, c.Nu))
    Wi = Wi * c.alpha_i
    # print("Wi:")
    # print("WoT\n", WoT)
    # print(Wi)Ds = np.zeros((MM*NN, Ny))
    Us = np.zeros((c.MM*c.NN, c.Nu))

    ### Wo
    Wo = np.zeros(c.Nh * c.Ny)
    Wo = Wo.reshape((c.Ny, c.Nh))
    Wo = Wo
    # print(Wo)

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

    rs = 1
    rs_prev = 0
    any_hs_change = True
    count=0
    m=0
    for n in range(c.NN * c.MM):
        theta = np.mod(n/c.NN,1) # (0,1)
        rs_prev = rs
        hs_prev = hs.copy()

        rs = p2s(theta,0)
        us = p2s(theta,Up[m])
        ds = p2s(theta,Dp[m])
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

        if rs==1: hc+=hs # デコードのためのカウンタ、ref.clockとhsのANDでカウントアップ
        count = count + 1

        # ref.clockの立ち上がり
        if rs_prev==0 and rs==1:
            hp = 2*hc/c.NN-1
            hc = np.zeros(c.Nh) #カウンタをリセット
            #ht = 2*hs-1 リファレンスクロック同期用ラッチ動作をコメントアウト
            yp = fy(Wo@hp)
            count=0
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
        Ds[n]=ds

    # 不連続な値の変化を検出する。
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
    invD = fyi(Dp)
    G = invD[c.MM0:, :]

    #print("Hp\n",Hp)
    #print("M\n",M)

    ### Ridge regression
    E = np.identity(c.Nh)
    TMP1 = inv(M.T@M + c.lambda0 * E)
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
    #if c.seed>=0:
    np.random.seed(int(c.seed))
    #np.random.seed(c.seed)

    generate_weight_matrix()

    ### generate data
    if c.dataset==1:
        MM1=300 # length of training data
        MM2=400 # length of test data
        D, U = generate_simple_sinusoidal(MM1+MM2)
    if c.dataset==2:
        MM1=300 # length of training data
        MM2=300 # length of test data
        D, U = generate_complex_sinusoidal(MM1+MM2)
    if c.dataset==3:
        MM1=1000 # length of training data
        MM2=1000 # length of test data
        D, U = generate_coupled_lorentz(MM1+MM2)
    D1 = D[0:MM1]
    U1 = U[0:MM1]
    D2 = D[MM1:MM1+MM2]
    U2 = U[MM1:MM1+MM2]

    ### training
    #print("training...")
    c.MM=MM1
    Dp = np.tanh(D1)
    Up = np.tanh(U1)
    train_network()

    ### test
    #print("test...")
    c.MM=MM2
    Dp = np.tanh(D2)
    Up = np.tanh(U2)
    test_network()

    ### evaluation
    sum=0
    for j in range(c.MM0,c.MM):
        sum += (Yp[j] - Dp[j])**2
    SUM=np.sum(sum)
    RMSE1 = np.sqrt(SUM/c.Ny/(c.MM-c.MM0))
    RMSE2 = 0
    print(RMSE1)

    c.RMSE1=RMSE1
    c.RMSE2=RMSE2
    c.cnt_overflow=cnt_overflow

    if c.plot: plot1()

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-config", type=str)
    a = ap.parse_args()

    c=Config()
    if a.config: c=common.load_config(a)
    execute()
    if a.config: common.save_config(c)

    print("asdf",c.RMSE1)
