import numpy as np 
from test_cbmrc9a_approximation2 import cbm_optimize
from test_esn_approximation2 import esn_optimize
import matplotlib.pyplot as plt 


"""
approximation task のdelay,logvで２次元マップを作るためのコード

    test_cbmrc9a_approximation2.py
    test_esn_approximation2.py

    cbmrc9a_approximation2.py
    esn_approximation2.py

    generate_data_sequence_approximation.py

    explorer 

    などを使用する。

"""

class Config1():
    def __init__(self):
        # columns, csv, id: データの管理のために必須の変数
        self.columns = None # 結果をCSVに保存する際のコラム
        self.csv = None # 結果を保存するファイル
        self.id  = None
        self.plot = True # 図の出力のオンオフ
        self.show = True # 図の表示（plt.show()）のオンオフ、explorerは実行時にこれをオフにする。
        self.savefig = True
        self.fig1 = "fig1.png" ### 画像ファイル名

        # config
        self.dataset=1
        self.seed:int=0 # 乱数生成のためのシード
        self.NN=256 # １サイクルあたりの時間ステップ
        self.MM=500 # サイクル数
        self.MM0 = 10 #

        self.Nu = 1   #size of input
        self.Nh = 300 #size of dynamical reservior
        self.Ny = 1   #size of output

        self.Temp=1.0
        self.dt=1.0/self.NN #0.01

        #sigma_np = -5
        self.alpha_i = 1
        self.alpha_r = 0.75
        self.alpha_b = 0.
        self.alpha_s = 2
        self.alpha0 = 1

        self.beta_i = 0.9
        self.beta_r = 0.1
        self.beta_b = 0.1

        self.lambda0 = 0.
        self.delay =1
        self.logv = 1
        
        # Results
        self.RMSE1=None
        self.NRMSE=None
        self.cnt_overflow = None 

    def change(self,logv,delay):
        self.logv = logv
        self.delay = delay


class Config2():
    def __init__(self):
        # columns, csv, id: データの管理のために必須の変数
        self.columns = None # 結果をCSVに保存する際のコラム
        self.csv = None # 結果を保存するファイル
        self.id  = None
        self.plot = True # 図の出力のオンオフ
        self.show = False # 図の表示（plt.show()）のオンオフ、explorerは実行時にこれをオフにする。
        self.savefig = False
        self.fig1 = "fig1.png" ### 画像ファイル名

        # config
        self.dataset=6
        self.seed:int=2 # 乱数生成のためのシード
        self.MM=500 # サイクル数
        self.MM0 = 10 #

        self.Nu = 1   #size of input
        self.Nh:int = 300#815 #size of dynamical reservior
        self.Ny = 1   #size of output


        #sigma_np = -5
        self.alpha_i = 0.02
        self.alpha_r = 0.52
        self.alpha_b = 0.

        self.alpha0 = 1#0.1
        self.alpha1 = 0#-5.8

        self.beta_i = 0.9
        self.beta_r = 1
        self.beta_b = 0.1

        self.lambda0 = 0.0

        self.delay = 1
        self.logv = 1

        # Results
        self.RMSE1=None
        self.NRMSE=None

    def change(self,logv,delay):
        self.logv = logv
        self.delay = delay

if __name__ == "__main__":

    logv = np.arange(-2,2,10)
    delay = np.arange(0,20,1,dtype=np.int)


    x = logv.shape[0]
    y = delay.shape[0]

    result = np.zeros((x,y))
    
    #Config2.change(Config2,logv = 0,delay= 1,)
#"""
    for j in range(y):              #遅延長
        for i in range(x):          #非線形性
            print("delay = {0}, logv = {1}".format(i,j))
            c2 = Config2()
            c2.change(logv = logv[i],delay= delay[j])
            c1 = Config1()
            c1.change(logv = logv[i],delay= delay[j])

            esn = esn_optimize(c2)

            cbm = cbm_optimize(c1)

            print(esn,cbm)
            diff = esn - cbm


            if diff>0:                #cbmが勝ったら = 1
                print("cbmの勝ち")
                result[i][j] = 1
                plt.scatter(j,i,marker = "o",label="cbm",color="b")
            elif diff<0:              #esnが勝ったら = -1
                print("esnの勝ち")
                result[i][j] = -1
                plt.scatter(j,i,marker = "x",label = "esn",color="r")
            else:
                plt.scatter(j,i,marker = "^",label="draw",color="k")

    plt.title("cbm vs esn on memory and nonlinearlity")
    plt.ylabel("delay")
    plt.xlabel("logv")
    plt.legend()
    plt.show()

#"""
