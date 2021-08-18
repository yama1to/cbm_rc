# Copyright (c) 2018-2021 Katori lab. All Rights Reserved
# NOTE:MG方程式をベンチマークに使用する。

import numpy as np
import scipy.linalg
from numpy.linalg import svd, inv, pinv
import matplotlib.pyplot as plt
import sys
import copy
from explorer import common


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

def flgt(h):
    return np.log(1/(1-h))

def p2s(theta,p):
    return np.heaviside( np.sin(np.pi*(2*theta-p)),1)

def run_network(MM,Ttf):
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
    Hp[m]=hp
    Yp[m]=yp
    for n in range(NN*MM):
        theta = np.mod(n/NN,1) # (0,1)
        rs_prev = rs
        hs_prev = hs.copy()

        rs = p2s(theta,0)
        #us = p2s(theta,Up[m])
        ds = p2s(theta,Dp[m])
        ys = p2s(theta,yp)

        sum = np.zeros(Nh)
        sum += alpha_s*rs # ラッチ動作を用いないref.clockと同期させるための結合
        #sum += alpha_s*(hs-rs)*ht # ref.clockと同期させるための結合
        #sum += Wi@(2*us-1) # 外部入力
        sum += Wr@(2*hs-1) # リカレント結合

        if n < NN*Ttf:
            sum += Wb@ds # teacher forcing
        else:
            sum += Wb@ys

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
            yp = fy(Wo@hp)
            #yp=fsgm(Wo@hp)
            count=0
            # record
            Hp[m+1]=hp
            Yp[m+1]=yp
            m+=1

        # record
        Rs[n]=rs
        Hx[n]=hx
        Hs[n]=hs
        Yx[n]=yx
        Ys[n]=ys
        #Us[n]=us
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

    run_network(T1,T1) # run netwrok with teacher forcing

    M = Hp[T0:, :]
    invD = fyi(Dp)
    G = invD[T0:, :]

    #print("Hp\n",Hp)
    #print("M\n",M)

    ### Ridge regression
    E = np.identity(Nh)
    TMP1 = inv(M.T@M + lambda0 * E)
    WoT = TMP1@M.T@G
    Wo = WoT.T
    #print("WoT\n", WoT)

def test_network():
    run_network(T2,T0)

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
    #ax.plot(Up)

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


def generate_data_sequence():
    D = np.zeros((MM, Ny))
    U = np.zeros((MM, Nu))
    cy = np.linspace(0, 1, Ny)
    cu = np.linspace(0, 1, Nu)
    for n in range(MM):
        t = 0.25 * n #0.5*n
        d = np.sin(t + cy) * 0.8
        # d=np.sin(t+c)*np.exp(-0.1*(t-10)**2)*0.5
        u = np.sin(t*0.5 + cu) * 0.8
        D[n, :] = d
        U[n, :] = u
    return (D, U)

def generate_data(MM):
    D = np.zeros(MM)
    #U = np.zeros((MM, Nu))
    #cy = np.linspace(0, 1, Ny)
    #cu = np.linspace(0, 1, Nu)
    for n in range(MM):
        t = 0.25 * n #0.5*n
        d = np.sin(t) * 0.8
        # d=np.sin(t+c)*np.exp(-0.1*(t-10)**2)*0.5
        #u = np.sin(t*0.5 + cu) * 0.8
        D[n] = d
        #U[n, :] = u
    return D

def generate_mackey_glass(tau,T1):
    '''
    tau: delay, typical value of tau = 17,
    T1: length
    '''
    delta = 0.1 #time constant
    T0 = int(tau / delta)
    T2 = T0+1000
    y = np.zeros(T2+T1)

    for n in range(T2+T1-1):
        if n < T0-1:
            y[n + 1] = 1.0 #np.random.uniform(0.0, 1.0)
        else:
            y[n + 1] = y[n] + delta * (0.2 * y[n - T0] / (1 + pow(y[n - T0],10)) - 0.1 * y[n])

    #print(y)
    return y[T2:]

def execute():
    global D,Ds,Dp,U,Us,Up,Rs,R2s
    global RMSE
    global T0,T1,T2

    np.random.seed(seed)
    generate_weight_matrix()

    T0 = 200 #100 # length of transient,
    T1 = 4000 #400 # length of training data
    Ntest = 2
    Nstep = 3
    interval = 50
    T2 = T0 + interval*(Nstep-1) # length of test data
    Tdata = T1 + T2*Ntest # total length of data (training and test data)

    #y = generate_data(Tdata)
    y = generate_mackey_glass(17, Tdata)

    DD = np.zeros((Tdata,1))
    DD[:,0]=np.tanh(y-1)

    ### train network
    #print("Train network")
    Dp = DD[:T1]
    train_network()
    #plot1()

    ### test network
    #print("Test network")
    SUM = np.zeros(Nstep)
    for i in range(Ntest):
        Dp = DD[T1 + T2*i : T1 + T2*(i+1)]
        test_network()
        for j in range(Nstep):
            SUM[j] += (Yp[T0-1+interval*j] - Dp[T0-1+interval*j])**2
        #print("SUM:",SUM[0],SUM[1],SUM[2],SUM[3],SUM[4],SUM[5])
    # mean squre error
    RMSE = np.sqrt(SUM/Ntest)
    #print(RMSE[0],RMSE[1],RMSE[2],RMSE[3],RMSE[4],RMSE[5])
    print(RMSE,count_gap)
    if display :
        plot1()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-config", type=str)
    a = ap.parse_args()

    c=Config()
    if a.config: c=common.load_config(a)
    config(c)
    execute()
    if a.config: common.save_config(c)
