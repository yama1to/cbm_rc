import numpy as np
import scipy.linalg
from numpy.linalg import svd, inv, pinv
import matplotlib.pyplot as plt

NN=25
MM=25
MM0 = 1

Nu = 2   #size of input
Nh = 25 #size of dynamical reservior
Ny = 2   #size of output

sigma_np = -5
alpha_r = 0.8
alpha_b = 0.8
alpha_i = 0.8
beta_r = 0.1
beta_b = 0.1
beta_i = 0.1
alpha0 = 0.7
tau = 2
lambda0 = 0.1

def generate_data_sequence():
    D = np.zeros((MM, Ny))
    U = np.zeros((MM, Nu))
    cy = np.linspace(0, 1, Ny)
    cu = np.linspace(0, 1, Nu)
    for n in range(MM):
        t = 0.1 * n
        d = np.sin(t + cy) * 0.8
        # d=np.sin(t+c)*np.exp(-0.1*(t-10)**2)*0.5
        u = np.sin(t*0.3 + cu) * 0.8
        D[n, :] = d
        U[n, :] = u
    return (D, U)

    #print("WoT\n", WoT)
def generate_s_sequence(p, u):
    s = np.zeros((MM*NN, u))
    for m in range(MM):
        for n in range(u):
            s[m*NN:int(NN*p[m][n])+m*NN][n] = 1
            s[int(NN*p[m][n])+m*NN:int(NN)+m*NN][n] = 0
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
    Wo = np.ones(MM * Ny)
    Wo = Wo.reshape((Ny, MM))
    Wo = Wo
    # print(Wo)

def update(hx, hs, m):
    for n in range(NN):
        for h in range(Nh):
            if hx[n+m*NN][h]>=1:
                hs[n+m*NN][h]=1
                hx[n+m*NN][h]=1
            elif hx[n+m*NN][h]<=0:
                hs[n+m*NN][h]=0
                hx[n+m*NN][h]=0

def fx(h):
    return np.tanh(h)

def fy(h):
    return np.tanh(h)

def fyi(h):
    #print("WoT\n", WoT)
    return np.arctanh(h)

def fr(h):
    return np.fmax(0, h)

def fsgm(h):
    return 1.0/(1.0+np.exp(-h))

def flgt(h):
    return np.log(1/(1-h))

def run_network(mode):
    global Hx, Hs, Hp, Y, Ys, Yp, Y
    Hp = np.zeros((MM, Nh))
    Hx = np.zeros((MM*NN, Nh))
    Hs = np.zeros((MM*NN, Nh))
    Yp = np.zeros((MM, Ny))
    Ys = np.zeros((MM*NN, Ny))

    for n in range(NN):
        h = np.random.randint(0, 2, Nh)
        Hx[n, :] = h
        Hs[n, :] = h
    A = np.ones((NN, Nh))
    sign = np.zeros((NN, Nh))
    for m in range(MM):
        for n in range(NN):
            sum = np.zeros(Nh)

            sum += Wi@Us[n+m*NN, :]
            sum += Wr@Hs[n+m*NN, :]
            if mode == 0:
                sum += Wb@Ys[n+m*NN, :]
            if mode == 1:  # teacher forcing
                sum += Wb@Ds[n+m*NN, :]
            sign[n, :] = A[n, :] - 2*Hs[n+m*NN, :]

        for n in range(NN):
            print(sign[n, :]*(1+np.exp(sign[n, :]*sum/NN)))
            Hx[n+m*NN, :] = Hx[n+m*NN, :] + sign[n, :]*(1+np.exp(sign[n, :]))
        update(Hs, Hx, m)
        Ys[m, :] = fy(Wo@Hs[m, :])

def train_network():
    global Wo

    run_network(1) # run netwrok with teacher forcing

    M = Hs[MM0:, :]
    invD = fyi(Ds)
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
    fig=plt.figure()
    ax1 = fig.add_subplot(4,1,1)
    ax1.cla()
    ax1.plot(Ds)
    ax2 = fig.add_subplot(4,1,2)
    ax2.cla()
    ax2.plot(Us)
    ax3 = fig.add_subplot(4,1,3)
    ax3.cla()
    ax3.plot(Hx)
    ax4 = fig.add_subplot(4,1,4)
    ax4.cla()
    ax4.plot(Hs)
    plt.show()

def execute():
    global D,Ds,Dp,U,Us,Up
    generate_weight_matrix()
    D, U = generate_data_sequence()
    Dp = fsgm(D)
    Up = fsgm(U)
    Ds = generate_s_sequence(Dp, Ny)
    Us = generate_s_sequence(Up, Nu)


    train_network()
    test_network()
    plot2()

if __name__ == "__main__":
    execute()
