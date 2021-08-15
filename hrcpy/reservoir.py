import numpy as np
from encode_decode import clock
from input import *
from data_generater import *
import scipy


class Reservoir:
    def __init__(self,N_x,U,step,density,rho,activation_func,
                alpha0,alpha1,alpha_r,alpha_s,beta_r,T,seed=0):
        """
        param N_x: リザバーのノード数
        param density: 内部状態の濃度
        param rho: スペクトル半径
        param activation_func: 活性化関数
        param leaking_rate: 漏れ率
        """
        self.seed = seed 
        self.N_x = N_x 

        self.step = step
        self.x = np.random.uniform(0,1,(U.shape[1]*self.step,N_x))
        self.s = np.zeros((U.shape[1]*self.step,N_x))

        self.activation= activation_func
        self.density = density
        self.rho = rho

        self.alpha0 = alpha0
        self.alpha1 = alpha1
        self.alpha_r = alpha_r
        self.beta_r = beta_r
        
        self.alpha_s = alpha_s
        self.W = self.Wrec()
        self.clock = clock(U.shape[1],self.step)
        self.T = T



    def Wrec(self,):
        #ランダムかつスパースな重み生成
        Wr0 = np.zeros(self.N_x * self.N_x)
        nonzeros = self.N_x * self.N_x * self.beta_r
        Wr0[0:int(nonzeros / 2)] = 1
        Wr0[int(nonzeros / 2):int(nonzeros)] = -1
        np.random.shuffle(Wr0)

        Wr0 = Wr0.reshape((self.N_x, self.N_x))
        v = scipy.linalg.eigvals(Wr0)
        lambda_max = max(abs(v))

        Wr = Wr0 / lambda_max * self.alpha_r
        E = np.identity(self.N_x)
        Wr = Wr + self.alpha0*E
        #Wr = Wr + alpha1

        Wr = Wr + self.alpha1/self.N_x

        #sp = np.max(abs(np.linalg.eigvals(Wr)))

        return Wr #(r_num,r_num)

    def update_r_s(self,t):
        """
        param r_x: 内部状態
        """
        r_s = self.s[t]
        r_x = self.x[t]
        #print(r_s.shape)
        for i in range(r_s.shape[0]):
            if r_x[i] >= 1:
                r_s[i] = 1
                r_x[i] = 1

            if r_x[i] <= 0:
                r_s[i] = 0
                r_x[i] = 0
        #print(r_s,r_x)
        return r_s,r_x

    def update_r_x(self,t,input):
        """
        1<=t<=39999
        dt = t/200
        """
        #dt = t/self.step

        I = self.W @ (2 * self.s[t] - 1) + input(t)
        #floor_t = np.floor(dt).astype(int)*self.step
        floor_t = t - t%self.step
        J = self.alpha_s * (self.s[t] -  self.clock[t]) * (2 * self.clock[floor_t] - 1)

        h = (1-2*self.s[t])*(I+J)
        #print(h)
        dx = (1-2*self.s[t])*(1+np.exp(h/self.T))/self.step
        #print(dx)
        self.x[t+1] = self.x[t] + dx
    
    def __call__(self,t,input):
        """
        param x: 更新後の内部状態
        """
        self.update_r_x(t,input)
        return self.update_r_s(t+1) #r_s,r_x


if __name__ == "__main__":


    N_u = 1
    N_x = 100
    leaningTime = 200
    step = 200
    density = 0.5
    rho = 0.5
    activation_func = "tanh"
    seed = 0
    
    T = 1

    alpha_i = 0.2
    beta_i  = 0.2
    alpha0  = 0.6
    alpha1  = 0.1
    alpha_r = 0.2
    alpha_s = 0.6
    beta_r  = 0.3

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
                        alpha_s=alpha_s,
                        beta_r=beta_r,
                        T = 1,
                        seed=seed)

    time = step * data.shape[1]
    #print(reservoir.s.shape)
    #print(reservoir.x.shape)
    #print(reservoir.clock.shape)
    for i in range(0,time-1):
        reservoir(i,input)
    
    
    s = reservoir.s
    x = reservoir.x
    c = reservoir.clock
    print(s.shape,x.shape,c.shape)
    plt.plot(s[:200,0])
    plt.plot(x[:200,0])
    #plt.plot(c[:2000])
    #plt.show()
#   plt.plot(reservoir.x[:time])
    plt.show()