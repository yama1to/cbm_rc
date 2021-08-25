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

"""def audio(wf):
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True) # このストリームに書き込むと音がなる

    chunk = 1024 # チャンク単位でストリームに出力し音声を再生
    wf.rewind() # ポインタを先頭にする
    data = wf.readframes(chunk)
    while data:
        stream.write(data)
        data = wf.readframes(chunk)
    stream.close()
    p.terminate()
"""
def generate_coch():
    np.random.seed(seed=0)

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


    #print(train_data.shape)
    #print(valid_data.shape)
    """
    train_data (250,12500) #(データ数,データ) データ数 = 0x25,1x25,2x25,...,9x25
    valid_data (250,12500) #(データ数,データ)

    """

    #"""

    calc = LyonCalc()

    waveform = train_data
    sample_rate = 12500
    train_coch = np.zeros((250,195,55))
    
    for i in range(250):
        c = calc.lyon_passive_ear(waveform[i], sample_rate, decimation_factor=64, ear_q=8, step_factor=0.35, tau_factor=3)
        train_coch[i] = c

    waveform = valid_data
    valid_coch = np.zeros((250,195,55))
    for i in range(250):
        c = calc.lyon_passive_ear(waveform[i], sample_rate, decimation_factor=86, ear_q=8, step_factor=0.35, tau_factor=3)
        valid_coch[i] = c
    
    #print(train_coch.shape,valid_coch.shape)

    print(train_coch.shape)
    idx = np.random.randint(0,100)
    coch = train_coch[idx]
    print(idx)
    print(coch.shape)
    fig=plt.figure()
    
    ax = fig.add_subplot(2,1,1)
    ax.cla()
    ax.imshow(coch.T)

    ax = fig.add_subplot(2,1,2)
    ax.cla()
    plt.plot(waveform[idx])
    plt.xlim(0,waveform[idx].shape[0])

    plt.savefig("test.png")
    plt.show()
    init = np.zeros((250,10))
    train_target = init
    train_target[:50][-1] = 1
    
    train_target[50:100][-2] = 1
    train_target[100:150][-3] = 1
    train_target[150:200][-5] = 1
    train_target[200:250][-4] = 1

    valid_target = train_target

    return train_coch,valid_coch ,train_target, valid_target

if __name__ == "__main__":
    t,v,tD,vD = generate_coch()
    print(t[0],v[10])
    print(tD == vD)
    tD = tD-1
    print(tD == vD)