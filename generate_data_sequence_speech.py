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
#import pyaudio

from lyon.calc import LyonCalc

def shuffle_samples(*args):
    # *argsで可変長引数を受け取る。変数argsにリストで格納される

    # unzipで複数配列のリスト -> 要素毎にまとめたタプルのリスト　に変換
    zipped = list(zip(*args))
    np.random.shuffle(zipped)

    # unzipして複数配列のリストの形に戻す
    shuffled = list(zip(*zipped))
    
    result = []
    # np.arrayに変換する処理
    for ar in shuffled:
        result.append(np.asarray(ar))
    return result

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

    # shap = (250,195,86)
    shap = (250,312,78)
    train_coch = np.zeros(shap)

    for i in range(250):
        c = calc.lyon_passive_ear(waveform[i], sample_rate, decimation_factor=40, ear_q=8, step_factor=0.25, tau_factor=2)
        train_coch[i] = c
        if save:
            file = "/home/yamato/Downloads/cbm_rc/coch_dir/train-fig"+str(i)+".png"
            save_coch(c,file)
        

    waveform = valid_data
    valid_coch = np.zeros(shap)

    for i in range(250):
        c = calc.lyon_passive_ear(waveform[i], sample_rate, decimation_factor=40, ear_q=8, step_factor=0.25, tau_factor=2)
        valid_coch[i] = c
        #
        if save:
            file = "/home/yamato/Downloads/cbm_rc/coch_dir/valid-fig"+str(i)+".png"
            save_coch(c,file)

    return train_coch,valid_coch

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
    train_data,valid_data = getwaves(train,valid,save = save)

    #generate cochlear
    train_coch,valid_coch = convert2cochlea(train_data,valid_data,save)

    train_coch /= np.max(train_coch)
    valid_coch /= np.max(valid_coch)
    #generate target
    shap = (250,312,10)
    collecting_target = np.zeros(shap)

    for i in range(shap[0]):
        collecting_target[i,:,i//25] = 1
    
    train_target = copy.copy(collecting_target)
    valid_target = copy.copy(collecting_target)

    #対応関係をそのままにデータをランダムにする
    if shuffle:
        train_coch,train_target = shuffle_samples(train_coch,train_target)
        valid_coch,valid_target = shuffle_samples(valid_coch,valid_target)

    return train_coch,valid_coch ,train_target, valid_target

if __name__ == "__main__":
    
    t,v,tD,vD = generate_coch(seed = 0,save=0,shuffle=True)
    print(t.shape,v.shape,tD.shape,vD.shape)
    print(np.sum(abs(tD-vD)))
    print(t[0].shape)
    for i in range(250):
        print(tD[i,0],vD[i,0])
        