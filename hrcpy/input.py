import matplotlib.pyplot as plt
import numpy as np
from data_generater import Data
from encode_decode import encode, decode
from  eval import RMSE

class Input:
    def __init__(self,N_u,N_x,u,alpha_i,beta_i,step,seed=0):
        '''
        param N_u: 入力次元
        param N_x: リザバーのノード数
        param input_scale: 入力スケーリング
        '''
        np.random.seed(seed=seed)
        
        self.u_s = encode(u,step)
        self.N_x = N_x
        self.N_u = N_u
        self.alpha_i = alpha_i
        self.beta_i = beta_i

        self.Win = self.Win()
        self.output = self.Win @ (2*self.u_s - 1)

        
        

    def Win(self,):
        Wi = np.zeros(self.N_x * self.N_u)
        tmp = self.N_x * self.N_u * self.beta_i
        Wi[0:int(tmp / 2)] = 1
        Wi[int(tmp / 2):int(tmp)] = -1
        np.random.shuffle(Wi)
        Wi = Wi.reshape((self.N_x, self.N_u))
        Wi = Wi * self.alpha_i
        return Wi

    def __call__(self,i):
        """
        param u: 入力ベクトル
        """
        #print(self.output.shape)
        return self.output[:,i]

if __name__ == '__main__':
    N_u = 1
    N_x = 200
    step = 200


    data = Data.sinwave(L=50,X=N_u,Y=200)
    
    input= Input(N_u = N_u,
                 N_x = N_x, 
                 u = data,
                 step = step,
                 alpha_i = 0.1,
                 beta_i = 0.1,
                 seed=0)
    

    x,y = data.shape

    u_s = encode(data,step)

    dec = decode(u_s,step)

    i = 0
    input_ = input(i)
    plt.plot(input_)
    plt.show()