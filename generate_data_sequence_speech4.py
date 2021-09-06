#katoriLab
# generate_data_sequence_speechでは３次元配列で作成していましたが、
# このコードではコクリアグラムを２次元配列で作成する
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
import soundfile as sf
from tqdm import tqdm_notebook as tqdm
#import pyaudio

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
    train_data = np.zeros((250,12500))
    valid_data = np.zeros((250,12500))
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
                wa = wa[1500:14000]
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
                wa = wa[1500:14000]

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
    input_num = 78
    t_num = 312
    data_num = 250

    train_coch = np.empty((input_num,data_num*t_num))

    for i in range(data_num):
        c = calc.lyon_passive_ear(waveform[i], sample_rate, decimation_factor=40, ear_q=8, step_factor=0.25, tau_factor=2)
        train_coch[:,i*t_num:(i+1)*t_num] = c.T
        if save:
            file = "/home/yamato/Downloads/cbm_rc/coch_dir/train-fig"+str(i)+".png"
            save_coch(c,file)

    waveform = valid_data
    valid_coch = np.empty((input_num,data_num*t_num))

    for i in range(data_num):
        c = calc.lyon_passive_ear(waveform[i], sample_rate, decimation_factor=40, ear_q=8, step_factor=0.25, tau_factor=2)

        valid_coch[:,i*t_num:(i+1)*t_num] = c.T
        #
        if save:
            file = "/home/yamato/Downloads/cbm_rc/coch_dir/valid-fig"+str(i)+".png"
            save_coch(c,file)

    return train_coch,valid_coch

def generate_target():
    shap = (250*312,10)
    collecting_target = np.zeros(shap)

    start = 0
    for i in range(250):
        collecting_target[start:start + 312,i//25] = 1
        start += 312 

    train_target = copy.copy(collecting_target)
    valid_target = copy.copy(collecting_target)

    return train_target,valid_target
    

def generate_coch(seed = 0,save=0,shuffle=True):
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

    





    return train_coch,valid_coch ,train_target, valid_target

if __name__ == "__main__":
    
    tD,vD = generate_target()
    print(tD.shape,vD.shape)
    t,v,tD,vD = generate_coch(seed = 0,save=0,shuffle=1)
    print(np.max(t),np.min(t),np.max(v),np.min(v))
    #print(t.shape,v.shape)
    # plt.plot(t.T)
    # plt.show()
    #for i in range(250):
    #    print(tD[i,0],vD[i,0])
        