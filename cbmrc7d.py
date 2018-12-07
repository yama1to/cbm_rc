# Copyright (c) 2018 Katori lab. All Rights Reserved
# NOTE:MG方程式をベンチマークに使用する。

import numpy as np
import scipy.linalg
from numpy.linalg import svd, inv, pinv
import matplotlib.pyplot as plt
import sys
import copy
from arg2x import *

NN=200
MM=300
MM0 = 50

T1 = 200
T0 = 5 # transient

Nu = 2   #size of input
Nh = 100 #size of dynamical reservior
Ny = 1   #size of output

Temp=1
dt=1.0/NN #0.01

#sigma_np = -5
alpha_i = 0.2
alpha_r = 0.2
alpha_b = 1.5

alpha0 = 0#0.1
alpha1 = 0#-5.8
alpha_s = 0.6

beta_i = 0.1
beta_r = 0.1
beta_b = 0.1

#tau = 2
lambda0 = 0.1

id = 0
ex = 'ex'
seed=0
display=1

def config():
    global ex,display,seed,id,NN,Nh,alpha_i,alpha_r,alpha_b,alpha_s,alpha0,alpha1,beta_i,beta_r,beta_b,Temp,lambda0
    args = sys.argv
    for s in args:
        ex      = arg2a(ex, 'ex=', s)
        display = arg2i(display,"display=",s)
        seed    = arg2i(seed,"seed=",s)
        id      = arg2i(id,"id=",s)

        NN      = arg2i(NN, 'NN=', s)
        Nh      = arg2i(Nh, 'Nh=', s)
        alpha_i = arg2f(alpha_i,"alpha_i=",s)
        alpha_r = arg2f(alpha_r,"alpha_r=",s)
        alpha_b = arg2f(alpha_b,"alpha_b=",s)
        alpha_s = arg2f(alpha_s,"alpha_s=",s)
        alpha0  = arg2f(alpha0,"alpha0=",s)
        alpha1  = arg2f(alpha1,"alpha1=",s)
        beta_i  = arg2f(beta_i,"beta_i=",s)
        beta_r  = arg2f(beta_r,"beta_r=",s)
        beta_b  = arg2f(beta_b,"beta_b=",s)
        Temp    = arg2f(Temp,"Temp=",s)
        lambda0 = arg2f(lambda0, 'lambda0=', s)

def output():
    str="%d,%d,%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n" \
    % (seed,id,NN,Nh,alpha_i,alpha_r,alpha_b,alpha_s,alpha0,alpha1,beta_i,beta_r,beta_b,Temp,lambda0,RMSE[0],RMSE[1],RMSE[2],count_gap)
    #print(str)
    filename= 'data_cbmrc7c_' + ex + '.csv'
    f=open(filename,"a")
    f.write(str)
    f.close()

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

def generate_weight_matrix():
    global Wr, Wb, Wo, Wi
    ### Wr
    Wr0 = np.zeros(Nh * Nh)
    nonzeros = Nh * Nh * beta_r
    Wr0[0:int(nonzeros / 2)] = 1
    Wr0[int(nonzeros / 2):int(nonzeros)] = -1
    np.random.shuffle(Wr0)
    Wr0 = Wr0.reshape((Nh, Nh))
    v = scipy.linalg.eigvals(Wr0)
    lambda_max = max(abs(v))
    #print("WoT\n", WoT)
    Wr = Wr0 / lambda_max * alpha_r
    E = np.identity(Nh)
    Wr = Wr + alpha0*E
    #Wr = Wr + alpha1

    Wr = Wr + alpha1/Nh
    #Wr = Wr -0.06#/Nh

    # print("lamda_max",lambda_max)
    # print("Wr:")
    # print(Wr)

    ### Wb
    Wb = np.zeros(Nh * Ny)
    Wb[0:int(Nh * Ny * beta_b / 2)] = 1
    Wb[int(Nh * Ny * beta_b / 2):int(Nh * Ny * beta_b)] = -1
    np.random.shuffle(Wb)
    Wb = Wb.reshape((Nh, Ny))
    Wb = Wb * alpha_b
    #print("Wb:",Wb)

    ### Wi
    Wi = np.zeros(Nh * Nu)
    Wi[0:int(Nh * Nu * beta_i / 2)] = 1
    Wi[int(Nh * Nu * beta_i / 2):int(Nh * Nu * beta_i)] = -1
    np.random.shuffle(Wi)
    Wi = Wi.reshape((Nh, Nu))
    Wi = Wi * alpha_i
    # print("Wi:")
    # print("WoT\n", WoT)
    # print(Wi)Ds = np.zeros((MM*NN, Ny))
    #Us = np.zeros((MM*NN, Nu))

    ### Wo
    Wo = np.zeros(Nh * Ny)
    Wo = Wo.reshape((Ny, Nh))
    Wo = Wo
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

def update_s(x,s,N):
    for i in range(N):
        if x[i]>=1:
            x[i]=1
            s[i]=1
        if x[i]<=0:
            x[i]=0
            s[i]=0
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
        hs_prev = hs.copy()
        update_s(hx,hs,Nh)

        # hs の立ち下がりで count の値を hc に保持する。
        #for i in range(Nh):
        #    if hs_prev[i]==1 and hs[i]==0:
        #        hc[i]=count
        #print(n,n%NN,l,hs_prev[0],hs[0],hc[0])
        #if m<3 or m>298:print("%3d %3d %d %d %3d %f"%(n,m,rs,rs_prev,count,theta))

        #ref.clockとhsのANDを取って1ならばカウントアップ
        for i in range(Nh):
            if rs==1 and hs[i]==1:
                hc[i]=hc[i] + 1

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
    config()
    execute()
    output()
