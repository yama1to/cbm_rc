import numpy as np
import scipy.linalg
from numpy.linalg import svd, inv, pinv
import matplotlib.pyplot as plt
import sys
from arg2x import *

NN=400
MM=200
MM0 = 10

Nu = 2   #size of input
Nh = 50 #size of dynamical reservior
Ny = 2   #size of output

Temp=1
dt=1.0/NN #0.01

#sigma_np = -5
alpha_i = 0.7
alpha_r = 0.1
alpha_b = 0.

alpha0 =  0.3

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
    global id,ex,seed,display,Nh,alpha_i,alpha_r,alpha_b,alpha0,beta_i,beta_r,beta_b,Temp,lambda0
    args = sys.argv
    for s in args:
        id      = arg2i(id,"id=",s)
        ex      = arg2a(ex, 'ex=', s)
        seed    = arg2i(seed,"seed=",s)
        display = arg2i(display,"display=",s)

        Nh      = arg2i(Nh, 'Nh=', s)
        alpha_i = arg2f(alpha_i,"alpha_i=",s)
        alpha_r = arg2f(alpha_r,"alpha_r=",s)
        alpha_b = arg2f(alpha_b,"alpha_b=",s)
        alpha0  = arg2f(alpha0,"alpha0=",s)
        beta_i  = arg2f(beta_i,"beta_i=",s)
        beta_r  = arg2f(beta_r,"beta_r=",s)
        beta_b  = arg2f(beta_b,"beta_b=",s)
        Temp    = arg2f(Temp,"Temp=",s)
        lambda0 = arg2f(lambda0, 'lambda0=', s)

def output():
    str="%d,%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n" \
    % (id,seed,Nh,alpha_i,alpha_r,alpha_b,alpha0,beta_i,beta_r,beta_b,Temp,lambda0,RMSE)
    #print(str)
    filename= 'data_cbmrc2_' + ex + '.csv'
    f=open(filename,"a")
    f.write(str)
    f.close()


def generate_data_sequence():
    D = np.zeros((MM, Ny))
    U = np.zeros((MM, Nu))
    cy = np.linspace(0, 1, Ny)
    cu = np.linspace(0, 1, Nu)
    for n in range(MM):
        t = 0.5 * n #0.5*n
        d = np.sin(t + cy) * 0.8
        # d=np.sin(t+c)*np.exp(-0.1*(t-10)**2)*0.5
        u = np.sin(t*0.5 + cu) * 0.8
        D[n, :] = d
        U[n, :] = u
    return (D, U)

    #print("WoT\n", WoT)
def generate_s_sequence(p, u):
    s = np.zeros((MM*NN, u))
    for m in range(MM):

        #for i in range(u):
        #s[ m*NN : m*NN + int(NN*p[m][0]) ][0] = 1
        pm=p[m]
        for i in range(u):
            for n in range(NN):
                if n < NN*pm[i] :
                    s[m*NN + n][i]=1

        #print(m,m*NN,m*NN+int(NN*p[m][0]),pm,s[m*NN])
    #for n in range(MM*NN):
    #    print(s[n])

    return s

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
    #return np.arctanh(h)
    return -np.log(1.0/h-1.0)
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

def run_network(mode):
    global Hx, Hs, Hp, Y, Yx, Ys, Yp, Y
    Hp = np.zeros((MM, Nh))
    Hx = np.zeros((MM*NN, Nh))
    Hs = np.zeros((MM*NN, Nh))
    Yp = np.zeros((MM, Ny))
    Yx = np.zeros((MM*NN, Ny))
    Ys = np.zeros((MM*NN, Ny))

    hsign = np.zeros(Nh)
    #hx = np.zeros(Nh)
    hx = np.random.uniform(0,1,Nh)
    hs = np.zeros(Nh)
    hc = np.zeros(Nh)

    ysign = np.zeros(Ny)
    yx = np.zeros(Ny)
    ys = np.zeros(Ny)
    yc = np.zeros(Ny)

    m=0
    for n in range(NN*MM):
        us = Us[n]
        ds = Ds[n]

        sum = np.zeros(Nh)
        sum += Wi@us
        sum += Wr@hs
        if mode == 0:
            sum += Wb@ys
        if mode == 1:  # teacher forcing
            sum += Wb@ds

        hsign = 1 - 2*hs
        hx = hx + hsign*(1.0+np.exp(hsign*sum/Temp))*dt
        update_s(hx,hs,Nh)
        hc += hs

        #print(n,n%NN,sum[0],hs[0])

        sum = np.zeros(Ny)
        sum += Wo@hx
        ysign = 1 - 2*ys
        yx = yx + ysign*(1.0+np.exp(ysign*sum/Temp))*dt
        update_s(yx,ys,Ny)
        yc += ys

        # compute duty ratio
        if n%NN == 0 :
            hp=hc/NN
            hc=np.zeros(Nh)
            yp=yc/NN
            yc=np.zeros(Ny)
            if m==0:
                hp=0.5
                yp=0.5
            #print(hp[0])
            #record
            Hp[m]=hp
            Yp[m]=yp
            m+=1

        # record
        Hx[n]=hx
        Hs[n]=hs
        Yx[n]=yx
        Ys[n]=ys


def train_network():
    global Wo

    run_network(1) # run netwrok with teacher forcing

    M = Hp[MM0:, :]
    invD = fyi(Dp)
    G = invD[MM0:, :]

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

def plot2():
    fig=plt.figure(figsize=(20, 12))
    Nr=4
    ax = fig.add_subplot(Nr,1,1)
    ax.cla()
    ax.set_title("Up")
    ax.plot(Up)

    ax = fig.add_subplot(Nr,1,2)
    ax.cla()
    ax.set_title("Hp")
    ax.plot(Hp)

    ax = fig.add_subplot(Nr,1,3)
    ax.cla()
    ax.set_title("Yp")
    ax.plot(Yp)

    ax = fig.add_subplot(Nr,1,4)
    ax.cla()
    ax.set_title("Dp")
    ax.plot(Dp)


    plt.show()

def plot3():
    fig=plt.figure(figsize=(20, 12))
    Nr=4

    ax = fig.add_subplot(Nr,1,1)
    ax.cla()
    ax.set_title("Us")
    ax.plot(Us)

    ax = fig.add_subplot(Nr,1,2)
    ax.cla()
    ax.set_title("Hx")
    ax.plot(Hx)

    ax = fig.add_subplot(Nr,1,3)
    ax.cla()
    ax.set_title("Ys")
    ax.plot(Ys)

    ax = fig.add_subplot(Nr,1,4)
    ax.cla()
    ax.set_title("Ds")
    ax.plot(Ds)

    plt.show()

def execute():
    global D,Ds,Dp,U,Us,Up
    global RMSE
    generate_weight_matrix()
    D, U = generate_data_sequence()
    Dp = fsgm(D)
    Up = fsgm(U)
    Ds = generate_s_sequence(Dp, Ny)
    Us = generate_s_sequence(Up, Nu)

    train_network()
    test_network()

    sum=0
    for j in range(MM0,MM):
        sum += (Yp[j] - Dp[j])**2
    SUM=np.sum(sum)
    RMSE=np.sqrt(SUM/Ny/(MM-MM0))

    print(RMSE)

    if display :
        plot2()
        #plot3()

if __name__ == "__main__":
    config()
    execute()
    output()
