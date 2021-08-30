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

def generate_coch(seed = 0):
    np.random.seed(seed=seed)

    num=["00","01","02","03","04","05","06","07","08","09"]
    person = ["f1","f2","m2","m3","m5"]
    times = ["set0","set1","set2","set3","set4","set5","set6","set7","set8","set9"]

    def num_split_data(num,person,times):
        all_data = list(itertools.product(num,person,times))
        all_data_join = [''.join(v) for v in all_data]
        return all_data_join

    data = np.array(num_split_data(num,person,times)).reshape(10,50)
    train = []
    valid = []

    for i in range(10):
        np.random.shuffle(data[i])
        train.append(data[i,:25])
        valid.append(data[i,25:])

    train = np.array(train)
    valid = np.array(valid)
    train_data = np.zeros((250,12500))
    valid_data = np.zeros((250,12500))

    #"""
    x = 0
    x1 = 0
    for i in range(10):
        #for file_name in train[i]:
        for j in range(train.shape[1]):
            file_name = train[i,j]
            file = "/home/yamato/Downloads/cbm_rc/ti-yamato/"+ str(file_name)+".wav"
            #print(file)
            with wave.open(file,mode='rb') as W:
                #print(W.getframerate())
                #print(W.getsampwidth())
                #print(W.getnchannels())
                W.rewind()
                buf = W.readframes(-1)  # read all

                #16bitごとに10進数
                wa = np.frombuffer(buf, dtype='int16')
                wa = wa[:12500]
                y = len(wa)
                train_data[x,:y] = wa[:y]
            
                x+= 1

        #"""
        #for file_name in train[i]:
        for j in range(valid.shape[1]):
            file_name = valid[i,j]
            file = "/home/yamato/Downloads/cbm_rc/ti-yamato/"+ str(file_name)+".wav"

            with wave.open(file,mode='rb') as W:
                W.rewind()
                buf = W.readframes(-1)  # read all

                #16bitごとに10進数
                wa = np.frombuffer(buf, dtype='int16')
                wa = wa[:12500]

                y = len(wa)
                valid_data[x1,:y] = wa[:y]

                x1+= 1


    calc = LyonCalc()

    waveform = train_data
    sample_rate = 12500

    shap = (250,195,86)
    train_coch = np.zeros(shap)
    for i in range(250):
        c = calc.lyon_passive_ear(waveform[i], sample_rate, decimation_factor=64, ear_q=8, step_factor=0.226, tau_factor=3)
        train_coch[i] = c

    waveform = valid_data

    valid_coch = np.zeros(shap)

    for i in range(250):
        c = calc.lyon_passive_ear(waveform[i], sample_rate, decimation_factor=64, ear_q=8, step_factor=0.226, tau_factor=3)
        valid_coch[i] = c
    

    idx = np.random.randint(0,100)

    shap2 = (250,195,10)
    collecting_target = np.zeros(shap2)

    for i in range(shap[0]):
        collecting_target[i,:,i//25] = 1
    
    train_target = copy.copy(collecting_target)

    valid_coch,valid_target = shuffle_samples(valid_coch,collecting_target)


    return train_coch,valid_coch ,train_target, valid_target


if __name__ == "__main__":

    t,v,tD,vD = generate_coch()
    print(t.shape,v.shape,tD.shape,vD.shape)
    print(np.sum(abs(tD-vD)))
    #$print(vD)

    """print(t[0],v[10])
    print(tD == vD)
    tD = tD-1
    print(tD == vD)"""