import numpy as np
from reservoir import Reservoir
from output import Output
from input import Input

class HypercubeReservoirComputing:
    def __init__(self,N_u,N_x,N_y,input_scale,
                density,rho,leaking_rate,U,D,step,
                activation_func=np.tanh,
                output_func = np.tanh,seed=0):
        """
        param N_u: 入力次元
        param N_x: リザバーのノード数
        param N_y: 出力次元
        param density: 内部状態の濃度
        param rho: スペクトル半径
        param activation_func: 活性化関数
        param output_func: 出力関数
        param seed: 乱数のシード

        """

        self.Input = Input(N_u,N_x,input_scale,seed=seed)
        self.Reservoir = Reservoir(N_x,U,step,density,rho,activation_func,leaking_rate,seed=seed)
        self.Output = Output(N_x,N_y,seed=seed)

        self.N_u = N_u
        self.N_x = N_x
        self.N_y = N_y

        self.r = np.zeros((N_x,U.shape[1]))
        self.y_prev = np.zeros(N_y)
        self.output_func = output_func


        def train(self,U,D,optimizer,step,T):
            """
            param U: 入力ベクトル dim x len
            param D: 目標出力ベクトル dim x len
            param optimizer: 最適化手法

            """
            

                #

            return 0

        def predict():
            return 0

        def ridge_regression():
            return 0
