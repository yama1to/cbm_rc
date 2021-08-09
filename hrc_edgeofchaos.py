import numpy as np
import scipy.linalg
from numpy.linalg import svd, inv, pinv
import matplotlib.pyplot as plt
import sys
import copy
from arg2x import *
import csv

NN=200
MM=2000
MM0 = 50

Nu = 2   #size of input
Nh = 100 #size of dynamical reservior
Ny = 2   #size of output

Temp=1
dt=1.0/NN #0.01

#sigma_np = -5
alpha_i = 0.2
alpha_r = 0.25
alpha_b = 0.

alpha0 = 0#0.1
alpha1 = 0#-5.8
alpha2 = 0.6

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
    global id,ex,seed,display,NN,Nh,alpha_i,alpha_r,alpha_b,alpha0,alpha1,alpha2,beta_i,beta_r,beta_b,Temp,lambda0

    args = sys.argv
    for s in args:
        id      = arg2i(id,"id=",s)
        ex      = arg2a(ex, 'ex=', s)
        seed    = arg2i(seed,"seed=",s)
        display = arg2i(display,"display=",s)

        NN      = arg2i(NN, 'NN=', s)
        Nh      = arg2i(Nh, 'Nh=', s)
        alpha_i = arg2f(alpha_i,"alpha_i=",s)
        alpha_r = arg2f(alpha_r,"alpha_r=",s)
        alpha_b = arg2f(alpha_b,"alpha_b=",s)
        alpha0  = arg2f(alpha0,"alpha0=",s)
        alpha1  = arg2f(alpha1,"alpha1=",s)
        alpha2  = arg2f(alpha2,"alpha2=",s)
        beta_i  = arg2f(beta_i,"beta_i=",s)
        beta_r  = arg2f(beta_r,"beta_r=",s)
        beta_b  = arg2f(beta_b,"beta_b=",s)
        Temp    = arg2f(Temp,"Temp=",s)
        lambda0 = arg2f(lambda0, 'lambda0=', s)

def output():
    str="%d,%d,%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n" \
    % (id,seed,NN,Nh,alpha_i,alpha_r,alpha_b,alpha0,alpha1,beta_i,beta_r,beta_b,Temp,lambda0,RMSE1,RMSE2)
    #print(str)
    filename= 'data_cbmrc3_' + ex + '.csv'
    f=open(filename,"a")
    f.write(str)
    f.close()


def generate_data_sequence():
    D = np.zeros((MM, Ny))
    U = np.zeros((MM, Nu))
    cy = np.linspace(0, 1, Ny)
    cu = np.linspace(0, 1, Nu)
    L = 50
    for n in range(MM):
        t = n #0.5*n
        d = np.sin(2*np.pi*t/L  + cy) * 0.8
        # d=np.sin(t+c)*np.exp(-0.1*(t-10)**2)*0.5
        u = np.sin(2*np.pi*t/L + cu) * 0.8
        D[n, :] = d
        U[n, :] = u
    return (D, U)


def generate_weight_matrix(rho):
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

    Wr = change_sp_radius(Wr,rho)
    get_linalg(Wr)
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
    # print("Wb:")
    # print(Wb)

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
    Us = np.zeros((MM*NN, Nu))

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
    return np.heaviside( np.sin(np.pi*(2*theta-p)),1) #np.tanh(U)


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
        #if n % 100==0:
            #print(n)
        theta = np.mod(n/NN,1) # (0,1)
        rs_prev = rs
        rs = p2s(theta,0)
        us = p2s(theta,Up[m])
        ds = p2s(theta,Dp[m])
        ys = p2s(theta,yp)

        sum = np.zeros(Nh)
        sum += alpha2*(hs-rs)*ht # ref.clockと同期させるための結合
        sum += Wi@(2*us-1) # 外部入力
        sum += Wr@(2*hs-1) # リカレント結合

        #if mode == 0:
        #    sum += Wb@ys
        #if mode == 1:  # teacher forcing
        #    sum += Wb@ds

        hsign = 1 - 2*hs
        hx = hx + hsign*(1.0+np.exp(hsign*sum/Temp))*dt
        #print(hx[0])
        hs_prev = hs.copy()
        update_s(hx,hs,Nh)

        # hs の立ち下がりで count の値を hc に保持する。
        for i in range(Nh):
            if hs_prev[i]==1 and hs[i]==0:
                hc[i]=count
        #print(n,n%NN,l,hs_prev[0],hs[0],hc[0])
        #if m<3 or m>298:print("%3d %3d %d %d %3d %f"%(n,m,rs,rs_prev,count,theta))

        count = count + 1

        # ref.clockの立ち上がり
        if rs_prev==0 and rs==1:
            hp = 2*hc/NN-1
            ht = 2*hs-1
            yp = fy(Wo@hp)
            #yp=fsgm(Wo@hp)
            count=0
            # record
            Hp[m]=hp
            Yp[m]=yp
            m+=1

        # record
        Rs[n]=rs #reference clock
        Hx[n]=hx #r_x
        Hs[n]=hs #r_s
        Yx[n]=yx #
        Ys[n]=ys #y
        Us[n]=us #
        Ds[n]=ds #



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
    #print(M.shape)
    #print(G.shape)
    TMP1 = inv(M.T@M + lambda0 * E)
    WoT = TMP1@M.T@G
    Wo = WoT.T
    #print("aaaaaaaaaaaaaaaaaaaaaaaaa"+str(Wo.shape))
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
    visualize_x_input(Up,Hp)
    get_linalg(Wr)

def execute(desired_sp):
    global D,Ds,Dp,U,Us,Up,Rs,R2s
    global RMSE1,RMSE2
    generate_weight_matrix(desired_sp)
    D, U = generate_data_sequence()
    Dp = np.tanh(D)
    Up = np.tanh(U)

    train_network()
    test_network()

    #plt.plot(list(range(600)),Rs[:600])
    """
    plt.plot(list(range(600)),Hx[:600])
    plt.plot(list(range(600)),Hs[:600])
    plt.show()"""

    sum=0
    for j in range(MM0,MM):
        sum += (Yp[j] - Dp[j])**2
    SUM=np.sum(sum)
    RMSE1 = np.sqrt(SUM/Ny/(MM-MM0))
    RMSE2 = 0
    #print(RMSE1)
    """
    if display :
        plot1()
    """
    #return RMSE1
    print (Hp.shape)
    return Hp[::50]


def visualize_x_input(input,r_x):
    print(r_x.shape)
    print(input.shape)

    
    plt.plot(input[:,0],r_x[:,0])
    #plt.scatter(input[:,0],r_x[:,0])
    plt.xlabel("input")
    plt.ylabel("r_x")
    plt.show()

    fig = plt.figure()
    axes= fig.subplots(5,6)
    for i in range(30):
        axes[i//6,i%6].plot(input[:,0],r_x[:,i])
        axes[i//6,i%6].set_xlabel("input")
        axes[i//6,i%6].set_ylabel("r_x")


    plt.show()
    

def get_linalg(Wr):
    eigv_list = np.linalg.eigvals(Wr)
    sp_radius = np.max(np.abs(eigv_list))
    print("スペクトル半径:"+str(sp_radius))

def change_sp_radius(Wr,rho):
    eigv_list = np.linalg.eig(Wr)[0]
    sp_radius = np.max(np.abs(eigv_list))

    Wr *= rho/sp_radius
    return Wr

def det_coef(n,k):
    r = np.zeros(k)
    cov = np.cov()
    var = np.var() * np.var()
    r = cov/var


if __name__ == "__main__":
    num = 20
    div = 10
    r = list(range(num+1))
    p =[]
    sp   =[]
    for i in r:
        i = i/(div) 
        config()
        p.append(execute(desired_sp=i))
        #print(p)
        #print("p:")
        #print(len(p),len(p[0]))
        sp.append(i)

        output()
    
    with open('edgeOfChaos.csv', "w") as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(p)):
            writer.writerow(p[i])

    
    for i in range(len(sp)):    #スペクトル半径
        for j in range(1):      #ノード
            for k in range(20): #十分学習時間が経過した後半20個
            #print(p[i][20+j])
                plt.scatter(i,p[i][20+k][j])
    plt.xlabel("sp")
    plt.ylabel("x")
    plt.title("edge of chaos")
    plt.show()
