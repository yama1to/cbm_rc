import numpy as np
from encode_decode import *
from data_generater import Data
from input import Input
from reservoir import Reservoir

class Output:
    def __init__(self,r,N_x,target,labmda,T0,fy = np.tanh,fyi=np.arctanh,seed=0):
        """
        param N_x: リザバーのノード数
        param N_y: 出力次元

        """
        self.fy = fy
        self.fyi = fyi
        np.random.seed(seed=seed)
        
        
        self.N_x = N_x
        self.labmda = labmda

        self.r = r
        self.T0 = T0

        self.Wout = self.setweight(target)

    def setweight(self,target):
        M = self.r[:,self.T0:].T        #M(data-T0,  N_x)
        G = self.fyi(target).T    #G(Tdata-T0 ,N_y)
        #print((M.T@M ).shape)
        #print((M.T@G ).shape)
        #print(M.shape,G.shape)
        tmp =np.linalg.inv(M.T@M + self.labmda * np.identity(self.N_x))
        #print(tmp.shape)
        Wout = tmp@M.T@G
        
        return Wout.T
        

    def __call__(self,):
        #print(self.Wout)
        #print(self.r)
        #print(self.Wout.shape ,self.r.shape)
        y = self.fy(self.Wout @ self.r)
        return y
    
if __name__ == "__main__":
    N_u = 1
    N_x = 100
    N_y = 1
    leaningTime = 200
    step = 200
    density = 0.5
    rho = 0.5
    activation_func = np.tanh
    leaking_rate = 0.1
    seed = 0
    T = 1
    T0 = 50
    
    T = 1

    lam = 0.1
    alpha_i = 0.2
    beta_i  = 0.2
    alpha0  = 0.6
    alpha1  = 0.1
    alpha_r = 0.2
    alpha_s = 0.6
    beta_r  = 0.3

    target = Data.sinwave(L=50,X=N_y,Y=leaningTime-T0)


    data = Data.sinwave(L=50,X=N_u,Y=leaningTime)

    input = Input(N_u = N_u,
                 N_x = N_x, 
                 u = data,
                 step = step,
                 alpha_i = alpha_i,
                 beta_i = beta_i,
                 seed=seed)

    reservoir = Reservoir(N_x,
                        data,
                        step,
                        density,
                        rho,
                        activation_func,
                        alpha0=alpha0,
                        alpha1 = alpha1,
                        alpha_r=alpha_r,
                        alpha_s = alpha_s,
                        beta_r=beta_r,
                        T = 1,
                        seed=seed)

    time = step * data.shape[1]

    for i in range(0,time-1):
        reservoir(i,input)
    s = reservoir.s
    #print(s.shape)
    r = decode(s.T,step)
    
    output = Output(r=r,
                    N_x=N_x,
                    target = target,
                    labmda=lam,
                    T0=T0)
    
    y = output()

    plt.plot(reservoir.x[:10000])
    plt.plot(reservoir.s[:10000])
    plt.show()
    """
    print(y)
    plt.plot(data[0],label="input")
    plt.plot(y[0],label="output")
    plt.legend()
    plt.show()
    """
