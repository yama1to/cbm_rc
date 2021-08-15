import numpy as np


class Output:
    def __init__(self,N_x,N_y,seed=0):
        """
        param N_x: リザバーのノード数
        param N_y: 出力次元

        """
        np.random.seed(seed=seed)
        self.Wout = np.random.uniform(-1,1,(N_x,N_y))

    def __call__(self,r_x):
        return 0
    def setweight(self, Wout):
        self.Wout = Wout 
