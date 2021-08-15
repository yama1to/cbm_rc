import numpy as np
from encode_decode import clock
from input import *

class Reservoir:
    def __init__(self,N_x,U,step,density,rho,activation_func,leaking_rate,T,seed=0):
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
        self.alpha = leaking_rate

        self.W = self.make_connection()
        self.clock = clock(U.shape[1],self.step)
        self.T = T



    def make_connection(self,):
        #ランダムかつスパースな重み生成
        w_rec_0 = np.zeros((self.N_x,self.N_x))
        ran = (self.density * (self.N_x **2))
        half = int(ran/2)
        w_rec_0[:half] = 1
        w_rec_0[half:int(ran)] = -1
        np.random.shuffle(w_rec_0)

        #スペクトル半径を決める
        value , _ = np.linalg.eig(w_rec_0)#　固有値
        w_rec = w_rec_0 * (self.rho/max(abs(value)))

        return w_rec #(r_num,r_num)

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
        dt = t/self.step

        I = self.W @ (2 * self.s[t] - 1) + input(t)
        J = self.alpha * (self.s[t] -  self.clock[t]) * (2 * self.s[np.floor(dt).astype(int)*self.step] - 1)

        h = (1-2*self.s[t])*(I+J)
        dx = (1-2*self.s[t])*(1+np.exp(h/self.T))
        self.x[t+1] = self.x[t] + dx

            #デコードしてrを求める
    
    def __call__(self,t,input):
        """
        param x: 更新後の内部状態
        """
        self.update_r_x(t,input)
        return self.update_r_s(t) #r_s,r_x


if __name__ == "__main__":


    N_u = 1
    N_x = 200
    step = 200
    density = 0.5
    rho = 0.5
    activation_func = "tanh"
    leaking_rate = 0.1
    seed = 0
    T = 1

    data = Data.sinwave(L=50,X=N_u,Y=200)

    input = Input(N_u = N_u,
                 N_x = N_x, 
                 u = data,
                 step = step,
                 input_scale= 1, 
                 seed=seed)

    reservoir = Reservoir(N_x,
                        data,
                        step,
                        density,
                        rho,
                        activation_func,
                        leaking_rate,
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
    #plt.plot(s[:2000,0])
    plt.plot(x[:2000,:])
    plt.plot(c[:2000])
    plt.show()
#    plt.plot(reservoir.x[:time])
#    plt.show()