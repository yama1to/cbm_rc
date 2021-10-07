import numpy as np 
from test_cbmrc9a_memory4 import cbm_optimize
from test_esn_memory4 import esn_optimize
import matplotlib.pyplot as plt 
from explorer import common 
import os 
from tqdm import tqdm 

"""
memory task のdelay,Nh で２次元マップを作るためのコード
    test_cbmrc9a_memory4.py
    test_esn_memory4.py

    cbmrc9a_memory4.py
    esn_memory4.py

    generate_data_sequence_memory.py

    explorer 
    などを使用する。

"""

class Config1():#cbm
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
        self.NN=256 # １サイクルあたりの時間ステップ
        self.MM=500 # サイクル数
        self.MM0 = 0 #

        self.Nu = 1   #size of input
        self.Nh:int = 300#815 #size of dynamical reservior
        self.Ny = 20   #size of output

        self.Temp=1
        self.dt=1.0/self.NN #0.01

        #sigma_np = -5
        self.alpha_i = 4.64
        self.alpha_r = 0.9
        self.alpha_b = 0.
        self.alpha_s = 5.91

        self.alpha0 = 0#0.1
        self.alpha1 = 0#-5.8

        self.beta_i = 0.67
        self.beta_r = 0.5
        self.beta_b = 0.1

        self.lambda0 = 0.

        self.delay = 20

        # ResultsX
        self.MC = None

        self.cnt_overflow=None



    def update(self,Nh,delay):
        self.Nh = Nh 
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
        self.MM0 = 0 #

        self.Nu = 1   #size of input
        self.Nh:int = 300#815 #size of dynamical reservior
        self.Ny = 20   #size of output


        #sigma_np = -5
        self.alpha_i = 1
        self.alpha_r = 0.9
        self.alpha_b = 0.

        self.alpha0 = 1#0.1
        self.alpha1 = 0#-5.8

        self.beta_i = 0.9
        self.beta_r = 1
        self.beta_b = 0.1

        self.lambda0 = 0.0

        self.delay = 10


        # Results
        self.MC = None



    def update(self,Nh,delay):
        self.Nh = Nh 
        self.delay = delay


if __name__ == "__main__":
    # save fiqure 
    common.prepare_directory("%s/memory_fig_dir" % os.getcwd())
    file_name = "memory_fig_dir/%s_memory" % common.string_now()


    def settingOptimize(flag):
        #optimize
        global iteration,population,samples,delay,Nh
        if flag:#本番環境
            iteration = 10
            population = 10
            samples = 3
            delay = np.array([20])
            Nh = np.arange(20,401,100,dtype = np.int)
            
        else:#動作確認
            iteration = 5
            population = 2
            samples = 1
            delay = np.array([10,20])
            Nh = np.array([300])

    settingOptimize(1)

    y = delay.shape[0]
    x = Nh.shape[0]

    cbm_mc = np.zeros(x)
    esn_mc = np.zeros(x)

    for i in range(y):
        for j in range(x):
            c1 = Config1()
            c2 = Config2()
            c1.update(Nh[j],delay[i])
            c2.update(Nh[j],delay[i])
            

            cbm_mc[j] = cbm_optimize(c1,iteration=iteration,population=population,samples=samples)
            esn_mc[j] = esn_optimize(c2,iteration=iteration,population=population,samples=samples)

        plt.plot(Nh,cbm_mc,marker = "o",label = "cbm")
        plt.plot(Nh,esn_mc,marker = "^",label = "esn")
        plt.title("optimizing MC, delay ={0}".format(delay[i]))
        plt.xlabel("Nh")
        plt.ylabel("memory capacity")
        plt.grid()
        plt.legend()
        plt.savefig(file_name+"_delay={0}.png".format(delay[i]))
        #plt.show()
        plt.clf()

            






    
