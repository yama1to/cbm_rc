import numpy as np

class Data:
    def sinwave(L,X,Y):
        """
        param L: 周期
        param X,Y: shape
        """
        sin = np.zeros((X,Y))
        for x in range(X):
            for y in range(Y):
                sin[x,y] = np.sin(2*np.pi*y/L)

        return sin 
