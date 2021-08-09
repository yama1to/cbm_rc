from test import W
import numpy as np

def phi(t):
    return np.heaviside(np.sin(2*np.pi*t),1)


class Input:
    def __init__(self,N_u,N_x, input_scale,seed=0):
        '''
        param N_u: 入力次元
        param N_x: リザバーのノード数
        param input_scale: 入力スケーリング
        '''

        np.random.seed(seed=seed)
        self.Win = np.random.uniform(-input_scale,input_scale,(N_x,N_u))

    def encode(self,u):
        """
        param u: 入力ベクトル
        """
        x,y = u.shape
        n=200
        u_s = np.zeros((x,y*n))
        for t in range(y*n):
            for udim in range(u.shape[0]):
                t_ = t/n
                u_s[0,t] = phi(t_ -u[0,int(np.floor(t_))]/2)

        return u_s

    def __call__(self,u):
        """
        param u: 入力ベクトル
        """
        u_s = self.encode(u)
        #return u_s
        return self.Win @ (2*u_s - 1)


class Reservoir:
    def __init__(self,N_x,density,rho,activation_func,leaking_rate,seed=0):
        self.seed = seed 
        self.N_x = N_x
        
        self.x =np.zeros(N_x)
        self.s =np.zeros(N_x)

        self.activation= activation_func
        self.density = density
        self.rho = rho
        self.alpha = leaking_rate

        self.W = self.make_connection() 

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

    def update_r_s(self,):
        """
        param r_x: 内部状態
        """
        r_s = self.r_s
        r_x = self.r_x
        
        if r_x >= 1:
            r_s = 1
            r_x = 1

        if r_x <= 1:
            r_s = 0
            r_x = 0

        return r_s,r_x

    def __call__(self,INPUT):
        I = self.W   @ (2*self.s - 1)
        J = INPUT
        T = 1
        self.x = self.x + (1-2*self.s)(1+np.exp((1-2*self.s)(I+J)/T))
        
        return self.update_r_s()




class Output:
    def __init__(self,N_x,N_y,seed=0):
        np.random.seed(seed=seed)
        self.Wout = np.random.uniform(-1,1,(N_x,N_y))

    def __call__(self,r_x):
        return 0
    def setweight(self, Wout):
        self.Wout = Wout 

    
class HypercubeReservoirComputing:
    def __init__(self,N_u,N_x,N_y,input_scale,
                density,rho,leaking_rate,activation_func=np.tanh,
                output_func = np.tanh,seed=0):

        self.Input = Input(N_u,N_x,input_scale,seed=seed)
        self.Reservoir = Reservoir(N_x,density,rho,activation_func,leaking_rate,seed=seed)
        self.Output = Output(N_x,N_y,seed=seed)

        self.N_u = N_u
        self.N_x = N_x
        self.N_y = N_y

        self.y_prev = np.zeros(N_y)
        self.output_func = output_func

        def train():
            return 0

        def predict():
            return 0




