#katoriLab
#
# speech4 のコクリアグラム生成時のパラメータ変化させる
#
from time import time
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy.io import loadmat
from tqdm.notebook import tqdm
import wave 
import itertools
import pandas as pd
import copy


from tqdm import tqdm_notebook as tqdm

from lyon.calc import LyonCalc

def num_split_data(num,person,times):
    all_data = list(itertools.product(num,person,times))
    all_data_join = [''.join(v) for v in all_data]
    return all_data_join

def save_wave_fig(wave,file):
    plt.plot(wave)
    plt.savefig(file)
    plt.clf()
    plt.close()

def save_coch(c,file):
    plt.imshow(c.T)
    plt.savefig(file)
    plt.clf()
    plt.close()

def getwaves(train,valid,save):
    x = 0
    x1 = 0
    train_data = np.zeros((250,19000))
    valid_data = np.zeros((250,19000))

    for i in range(10):
        for j in range(train.shape[1]):
            file_name = train[i,j]
            file = "/home/yamato/Downloads/cbm_rc/ti-yamato/"+ str(file_name)+".wav"
            #
            with wave.open(file,mode='r') as W:
                W.rewind()
                buf = W.readframes(-1)  # read allA
                #16bitごとに10進数
                wa = np.frombuffer(buf, dtype='int16')
                y = len(wa)
                train_data[x,:y] = wa[:y]
                
                if save:
                    save_file = "/home/yamato/Downloads/cbm_rc/fig_dir/"+ str(file_name)+".wav"
                    save_wave_fig(wa,save_file)
                
                x+= 1
        for j in range(valid.shape[1]):
            file_name = valid[i,j]
            file = "/home/yamato/Downloads/cbm_rc/ti-yamato/"+ str(file_name)+".wav"
            #
            with wave.open(file,mode='r') as W:
                W.rewind()
                buf = W.readframes(-1)  # read all

                #16bitごとに10進数
                wa = np.frombuffer(buf, dtype='int16')
                y = len(wa)
                valid_data[x1,:y] = wa[:y]

                if save:
                    save_file = "/home/yamato/Downloads/cbm_rc/fig_dir/"+ str(file_name)+".wav"
                    save_wave_fig(wa,save_file)
                x1+= 1

    return train_data,valid_data
        
def convert2cochlea(train_data,valid_data,save):

    calc = LyonCalc()

    waveform = train_data
    sample_rate = 12500
    train_coch = np.empty((input_num,data_num*t_num))

    for i in range(data_num):                                               #64                                 #3
        c = calc.lyon_passive_ear(waveform[i], sample_rate, decimation_factor=375, ear_q=8, step_factor=1, tau_factor=3)#
        c = c*10**4
        plt.plot(c)
        plt.show()
        print(c.shape)
        train_coch[:,i*t_num:(i+1)*t_num] = c.T
        
        if save:
            file = "/home/yamato/Downloads/cbm_rc/coch_dir/train-fig"+str(i)+".png"
            save_coch(c,file)

    waveform = valid_data
    valid_coch = np.empty((input_num,data_num*t_num))

    for i in range(data_num):
        c = calc.lyon_passive_ear(waveform[i], sample_rate, decimation_factor=375, ear_q=6, step_factor=None, tau_factor=3)

        valid_coch[:,i*t_num:(i+1)*t_num] = c.T
        #
        if save:
            file = "/home/yamato/Downloads/cbm_rc/coch_dir/valid-fig"+str(i)+".png"
            save_coch(c,file)

    return train_coch,valid_coch

def generate_target():
    shap = (250*t_num,10)
    collecting_target = np.zeros(shap)

    start = 0
    for i in range(250):
        collecting_target[start:start + t_num,i//25] = 1
        start += t_num 

    train_target = copy.copy(collecting_target)
    valid_target = copy.copy(collecting_target)

    return train_target,valid_target
    

def generate_coch(load=1,seed = 0,save=0,shuffle=True):  
    global data_num,t_num,input_num,SHAPE

    input_num = 77
    t_num = 50
    data_num = 250
    SHAPE = (data_num,t_num,input_num)

    if load:
        train_coch   = np.load("generate_cochlear_speech4train_coch.npy")
        valid_coch   = np.load("generate_cochlear_speech4valid_coch.npy")
        train_target = np.load("generate_cochlear_speech4train_target.npy")
        valid_target = np.load("generate_cochlear_speech4valid_target.npy")

        return train_coch,valid_coch ,train_target, valid_target, (SHAPE)
    np.random.seed(seed=seed)

    #file name
    num=["00","01","02","03","04","05","06","07","08","09"]
    person = ["f1","f2","m2","m3","m5"]
    times = ["set0","set1","set2","set3","set4","set5","set6","set7","set8","set9"]

    data = np.array(num_split_data(num,person,times)).reshape(10,50)
    train = np.empty((0,25))
    valid = np.empty((0,25))

    #numについてシャッフルしランダムに25ずつ分割する
    for i in range(10):
        np.random.shuffle(data[i])
        train = np.vstack((train,data[i,:25]))
        valid = np.vstack((valid,data[i,25:]))
    
    # generate wave
    print("~ generate wave ~")
    train_data,valid_data = getwaves(train,valid,save = save)

    #generate cochlear
    print("~ generate cochlear ~")
    train_coch,valid_coch = convert2cochlea(train_data,valid_data,save)

    #generate target
    print("~ generate target ~")
    train_target,valid_target = generate_target() 

    train_coch = train_coch.T
    valid_coch = valid_coch.T

    return train_coch,valid_coch ,train_target, valid_target, (SHAPE)

def save_data():
    file = "generate_cochlear_speech4"
    np.save(file+"train_coch",arr=t,)
    np.save(file+"valid_coch",arr=v,)
    np.save(file+"train_target",arr=tD,)
    np.save(file+"valid_target",arr=vD,)

if __name__ == "__main__":
    
    #tD,vD = generate_target()
    #print(tD.shape,vD.shape)
    t,v,tD,vD ,s= generate_coch(load = 0,seed = 0,save=0,shuffle=1)
    print(t)
    
    save_data()
    #print(np.max(t),np.min(t),np.max(v),np.min(v))
