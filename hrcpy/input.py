import matplotlib.pyplot as plt
import numpy as np
from data_generater import Data
from encode_decode import encode, decode
from  eval import RMSE

class Input:
    def __init__(self,N_u,N_x,u,step,input_scale,seed=0):
        '''
        param N_u: 入力次元
        param N_x: リザバーのノード数
        param input_scale: 入力スケーリング
        '''
        np.random.seed(seed=seed)
        self.Win = np.random.uniform(-input_scale,input_scale,(N_x,N_u))
        self.u_s = encode(u,step)
        self.output = self.Win @ (2*self.u_s - 1)

    def __call__(self,i):
        """
        param u: 入力ベクトル
        """

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
                 input_scale= 1, 
                 seed=0)
    

    x,y = data.shape

    u_s = encode(data,step)

    dec = decode(u_s,step)

    i = 0
    input_ = input(i)
    plt.plot(input_)
    plt.show()