# Copyright (c) 2018-2021 Katori lab. All Rights Reserved
"""
NOTE: cbm_rc　時系列生成タスク　
cbmrc6f2.pyを改変
latch動作
デコードの修正
"""

import argparse
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import copy
import time
from explorer import common
from generate_data_sequence import *
from generate_matrix import *
from tqdm import tqdm

class Cbmrc:
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
        self.MM=2000 # サイクル数
        self.MM0 = 100 #

        self.Nu = 1   #size of input
        self.Nh = 300 #size of dynamical reservior
        self.Ny = 20   #size of output

        self.Temp=1.0
        self.dt=1.0/self.NN #0.01

        #sigma_np = -5
        self.alpha_i = 0.2
        self.alpha_r = 0.25
        self.alpha_b = 0.
        self.alpha_s = 0.6

        self.beta_i = 0.1
        self.beta_r = 0.5
        self.beta_b = 0.1

        self.lambda0 = 0.0

        # Results
        self.RMSE1=None
        self.RMSE2=None
        self.cnt_overflow=None
        
        # 初期化


    def generate_weight_matrix(self,):
        global Wr, Wb, Wo, Wi
        Wr = generate_random_matrix(self.Nh,self.Nh,self.alpha_r,self.beta_r,distribution="one",normalization="sr")
        Wb = generate_random_matrix(self.Nh,self.Ny,self.alpha_b,self.beta_b,distribution="one",normalization="none")
        Wi = generate_random_matrix(self.Nh,self.Nu,self.alpha_i,self.beta_i,distribution="one",normalization="none")
        Wo = np.zeros(self.Nh * self.Ny).reshape(self.Ny, self.Nh)

        
    def fit(self,X_train,y_train):
        global Wo,G
        self.generate_weight_matrix()
        self.run_network(X_train,y_train)
        
        M = Hp[self.MM0:, :]
        invD = self.fyi(y_train)
        G = invD[self.MM0:, :]

        #print("Hp\n",Hp)
        #print("M\n",M)

        ### Ridge regression
        if self.lambda0 == 0:
            Wo = np.dot(G.T,np.linalg.pinv(M).T)
            #print("a")
        else:
            E = np.identity(self.Nh)
            TMP1 = np.linalg.inv(M.T@M + self.lambda0 * E)
            WoT = TMP1@M.T@G
            Wo = WoT.T
        return 

    def predict(self,X_test):
        self.run_network(X_test)
        return Yp[self.MM0:],G

    def fy(self,h):
        return np.tanh(h)

    def fyi(self,h):
        return np.arctanh(h)

    def p2s(self,theta,p):
        return np.heaviside( np.sin(np.pi*(2*theta-p)),1)
            
    def run_network(self,X,y = None):
        global Hx, Hs, Hp, Y, Yx, Ys, Yp, Y, Us, Ds,Rs
        Hp = np.zeros((self.MM, self.Nh))
        Hx = np.zeros((self.MM*self.NN, self.Nh))
        Hs = np.zeros((self.MM*self.NN, self.Nh))
        hsign = np.zeros(self.Nh)
        hx = np.zeros(self.Nh)
        #hx = np.random.uniform(0,1,self.Nh) # [0,1]の連続値
        hs = np.zeros(self.Nh) # {0,1}の２値
        hs_prev = np.zeros(self.Nh)
        hc = np.zeros(self.Nh) # ref.clockに対する位相差を求めるためのカウント
        hp = np.zeros(self.Nh) # [-1,1]の連続値
        ht = np.zeros(self.Nh) # {0,1}

        Yp = np.zeros((self.MM, self.Ny))
        Yx = np.zeros((self.MM*self.NN, self.Ny))
        Ys = np.zeros((self.MM*self.NN, self.Ny))
        #ysign = np.zeros(Ny)
        yp = np.zeros(self.Ny)
        yx = np.zeros(self.Ny)
        ys = np.zeros(self.Ny)
        #yc = np.zeros(Ny)

        Us = np.zeros((self.MM*self.NN, self.Nu))
        Ds = np.zeros((self.MM*self.NN, self.Ny))
        Rs = np.zeros((self.MM*self.NN, 1))

        rs = 1
        any_hs_change = True
        count =0
        m = 0
        for n in tqdm(range(self.NN * self.MM)):
            theta = np.mod(n/self.NN,1) # (0,1)
            rs_prev = rs
            hs_prev = hs.copy()

            rs = self.p2s(theta,0)# 参照クロック
            us = self.p2s(theta,X[m]) # エンコードされた入力
            #ds = self.p2s(theta,y[m]) #
            ys = self.p2s(theta,yp)

            sum = np.zeros(self.Nh)
            #sum += self.alpha_s*rs # ラッチ動作を用いないref.clockと同期させるための結合
            sum += self.alpha_s*(hs-rs)*ht # ref.clockと同期させるための結合
            sum += Wi@(2*us-1) # 外部入力
            sum += Wr@(2*hs-1) # リカレント結合

            #if mode == 0:
            #    sum += Wb@ys
            #if mode == 1:  # teacher forcing
            #    sum += Wb@ds

            hsign = 1 - 2*hs
            hx = hx + hsign*(1.0+np.exp(hsign*sum/self.Temp))*self.dt
            hs = np.heaviside(hx+hs-1,0)
            hx = np.fmin(np.fmax(hx,0),1)

            hc[(hs_prev == 1)& (hs==0)] = count
            
            # ref.clockの立ち上がり
            if rs_prev==0 and rs==1:
                hp = 2*hc/self.NN-1 # デコード、カウンタの値を連続値に変換
                hc = np.zeros(self.Nh) #カウンタをリセット
                ht = 2*hs-1 #リファレンスクロック同期用ラッチ動作をコメントアウト
                yp = Wo@hp
                # record    
                Hp[m]=hp
                Yp[m]=yp
                count = 0
                m += 1

            #境界条件
            if n == (self.NN * self.MM-1):
                hp = 2*hc/self.NN-1 # デコード、カウンタの値を連続値に変換
                yp = Wo@hp
                # record
                Hp[m]=hp
                Yp[m]=yp

            count += 1
            any_hs_change = np.any(hs!=hs_prev)

            if self.plot:
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
        for m in range(2,self.MM-1):
            tmp = np.sum( np.heaviside( np.fabs(Hp[m+1]-Hp[m]) - 0.6 ,0))
            cnt_overflow += tmp
            #print(tmp)

if __name__ == "__main__":
    cbm = Cbmrc()
    u,d = generate_white_noise(delay_s=20,T=cbm.MM,dist="uniform")
    u,d = u/5,d/5
    
    cbm.fit(u,d)
    y,d = cbm.predict(u)
    print(y.shape,d.shape)
    sum=0
    for j in range(cbm.MM-cbm.MM0):
        sum += (y[j] - d[j])**2

    SUM=np.sum(sum)
    RMSE1 = np.sqrt(SUM/cbm.Ny/(cbm.MM-cbm.MM0))
    RMSE2 = 0
    print (RMSE1)

    DC = np.zeros(20)
    for k in range(20):
        corr = np.corrcoef(np.vstack((d.T[k, k:], y.T[k, k:])))   #相関係数
        DC[k] = corr[0, 1] ** 2                                     #決定係数 = 相関係数 **2

    MC = np.sum(DC)
    plt.plot(DC)
    plt.show()
