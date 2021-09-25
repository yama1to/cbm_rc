import numpy as np 
from test_cbmrc9a_approximation2 import cbm_optimize
from test_esn_approximation2 import esn_optimize
import matplotlib.pyplot as plt 
from explorer import common 
import os 
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
        self.f = np.sin 

        # Results
        self.RMSE1=None
        self.NRMSE=None
        self.cnt_overflow = None 

    def update(self,logv,delay,f):
        self.logv = logv
        self.delay = delay
        self.f = f


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
        self.f = np.sin

        # Results
        self.RMSE1=None
        self.NRMSE=None

    def update(self,logv,delay,f):
        self.logv = logv
        self.delay = delay
        self.f = f

if __name__ == "__main__":
    # save fiqure 
    common.prepare_directory("%s/trade-off_fig_dir" % os.getcwd())
    file_name = "trade-off_fig_dir/%s_trade-off.png" % common.string_now()

    # setting for production
    #logv = np.arange(-2,2,0.5)
    #delay = np.arange(0,20,10,dtype=np.int)
    #f = list(np.sin,np.tan,lambda x: x(1-x**2))

    # test
    logv = np.arange(-2,2,4)
    delay = np.arange(0,20,20,dtype=np.int)
    f = np.array([np.sin])

    x = logv.shape[0]
    y = delay.shape[0]
    z = f.shape[0]

    result = np.zeros((x,y))
    
    #optimize
    iteration = 1
    population = 1
    samples = 1

    fig = plt.figure()

    

#"""
    for k in range(z):                  #非線形関数
        ax = fig.add_subplot(130+k+1)

        for j in range(y):              #遅延長
            for i in range(x):          #非線形性
                print("delay = {0}, logv = {1}".format(i,j))
                c2 = Config2()
                c2.update(logv = logv[i],delay= delay[j],f = f[k])
                c1 = Config1()
                c1.update(logv = logv[i],delay= delay[j],f = f[k])

                esn = esn_optimize(c2,iteration,population,samples)

                cbm = cbm_optimize(c1,iteration,population,samples)

                per = esn/cbm
                print(per)

                if per>1.05:                #cbmが勝ったら = 1
                    print("cbmの勝ち")
                    result[i][j] = 1
                    ax.scatter(j,i,marker = "o",label="cbm",color="b")
                elif per<0.95:              #esnが勝ったら = -1
                    print("esnの勝ち")
                    result[i][j] = -1
                    ax.scatter(j,i,marker = "x",label = "esn",color="r")
                else:
                    ax.scatter(j,i,marker = "^",label="draw",color="k")


        ax.title.set_text("%s" % str(f[k]))
        ax.set_ylabel("delay")
        ax.set_xlabel("logv")
        ax.legend()
    
    fig.tight_layout()

    fig.savefig(file_name)
    fig.show()

#"""
