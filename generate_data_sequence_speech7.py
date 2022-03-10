#katoriLab
#
# speech7 のコクリアグラム生成時のパラメータ変化させる
#
from time import time
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy.io import loadmat
import wave 
import itertools
import pandas as pd
import copy


from tqdm import tqdm

from lyon.calc import LyonCalc

global cwd
cwd = os.getcwd() # ~/cbmrc/
matrix_dir = "./speech7_matrix_dir/"


def num_split_data(num,person,times):
    all_data = list(itertools.product(num,person,times))
    all_data_join = [''.join(v) for v in all_data]
    return all_data_join

def save_wave_fig(wave,file_name):
    save_file = cwd + "/fig_dir/"+ str(file_name)+".png"
    plt.plot(wave)
    plt.savefig(save_file)
    plt.clf()
    plt.close()

def save_coch(c,file):
    plt.imshow(c.T)
    plt.savefig(file)
    plt.clf()
    plt.close()


def cut(wave):
    len_output=10000 # 出力波形の長さ
    #print(len(wave))
    
    count = 0
    start=0
    for i in range(len(wave)):
        if np.fabs(wave[i]) > 200:
            count += 1
            if count==10:
                start = i
                break 

    start = i-2500
    start = max(start,0)
    end = min(len_output+start,wave.shape[0])
    output = np.zeros(len_output)

    output[:end-start] = wave[start:end]
    return output

def loadwave(file_name):
    file = cwd+"/ti-yamato/"+ str(file_name)+".wav"
    #
    with wave.open(file,mode='r') as W:
        W.rewind()
        buf = W.readframes(-1)  # read allA
        #16bitごとに10進数
        wa = np.frombuffer(buf,dtype='int16').astype(np.float64)
    wa = cut(wa)
    return wa


def getwaves(train,valid,save=0,load=0):
    if load:
        train_data = np.load(matrix_dir+"train_data_wave.npy")
        valid_data = np.load(matrix_dir+"valid_data_wave.npy")
    if not load:
        x = 0
        x1 = 0
        len = 10000
        train_data = np.zeros((250,len))
        valid_data = np.zeros((250,len))

        for i in range(10):
            for j in range(train.shape[1]):
                file_name = train[i,j]
                train_data[x] = loadwave(file_name=file_name)

                if save:
                    save_wave_fig(train_data[x],file_name)
                    
                x+= 1
            for j in range(valid.shape[1]):
                file_name = valid[i,j]
                valid_data[x1] = loadwave(file_name)

                if save:
                    save_wave_fig(valid_data[x1],file_name)
                x1+= 1
    return train_data,valid_data
        
def convert2cochlea(train_data,valid_data,save):

    calc = LyonCalc()

    waveform = train_data
    sample_rate = 12500
    train_coch = np.empty((input_num,data_num*t_num))


    for i in range(data_num):
        #plt.plot(waveform[i])
        #plt.show()
                                                                        #64                                 #3
        c = calc.lyon_passive_ear(waveform[i], sample_rate, decimation_factor=200, ear_q=8,step_factor=0.226, tau_factor=3)#
        #print(c.shape)
        # plt.plot(c)
        # plt.show()
        train_coch[:,i*t_num:(i+1)*t_num] = c.T
        
        if save:
            file = cwd+"/coch_dir/train-fig"+str(i)+".png"
            save_coch(c,file)

    waveform = valid_data
    valid_coch = np.empty((input_num,data_num*t_num))

    for i in range(data_num):
        c = calc.lyon_passive_ear(waveform[i], sample_rate, decimation_factor=200, ear_q=8, step_factor=0.226, tau_factor=3)
        
        valid_coch[:,i*t_num:(i+1)*t_num] = c.T
        #
        if save:
            file = cwd + "/coch_dir/valid-fig"+str(i)+".png"
            save_coch(c,file)

    return train_coch,valid_coch

def generate_target():
    collecting_target = np.zeros((data_num*t_num,10))

    start = 0
    for i in range(250):
        collecting_target[start:start + t_num,i//25] = 1
        start += t_num 

    train_target = copy.copy(collecting_target)
    valid_target = copy.copy(collecting_target)

    return train_target,valid_target
    

def generate_coch(load=0,seed = 0,save_arr=1,save=0,shuffle=True):  
    global data_num,t_num,input_num,SHAPE

    input_num = 86
    t_num = 50
    data_num = 250
    SHAPE = (data_num,t_num,input_num)
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
    train_data,valid_data = getwaves(train,valid,save = save,load = load)

    if save_arr:
        np.save(matrix_dir+"train_data_wave",arr=train_data)
        np.save(matrix_dir+"valid_data_wave",arr=valid_data)

    #generate cochlear
    print("~ generate cochlear ~")
    train_coch,valid_coch = convert2cochlea(train_data,valid_data,save)

    #generate target
    print("~ generate target ~")
    train_target,valid_target = generate_target() 

    train_coch = train_coch.T
    valid_coch = valid_coch.T

    valid_coch2 = valid_coch.reshape(SHAPE)
    valid_target2 = valid_target.reshape((250,50,10))
    #print(valid_coch2.shape,valid_target2.shape)
    #valid シャッフル
    valid_coch2,valid_target2 = shuffle_samples(valid_coch2,valid_target2)
    valid_coch = valid_coch2.reshape((data_num*t_num,input_num))
    valid_target = valid_target2.reshape((data_num*t_num,10))

    if save_arr:
        save_data(train_coch,valid_coch ,train_target, valid_target)

    return train_coch,valid_coch ,train_target, valid_target, (SHAPE)

def save_data(t,v,tD,vD):
    file = matrix_dir+"generate_cochlear_speech7"
    np.save(file+"train_coch",arr=t,)
    np.save(file+"valid_coch",arr=v,)
    np.save(file+"train_target",arr=tD,)
    np.save(file+"valid_target",arr=vD,)

def load_datasets():
    SHAPE = (250,50,86)
    fname = matrix_dir+"generate_cochlear_speech7"
    train_coch   = np.load(fname+"train_coch.npy")
    valid_coch   = np.load(fname+"valid_coch.npy")
    train_target = np.load(fname+"train_target.npy")
    valid_target = np.load(fname+"valid_target.npy")


    return train_coch,valid_coch ,train_target, valid_target, SHAPE

# 配列X, yを与え、対応を崩さずにシャッフルする
def shuffle_samples(X, y):
    order = np.arange(X.shape[0])
    np.random.shuffle(order)
    X_result = np.zeros(X.shape)
    y_result = np.zeros(y.shape)
    for i in range(X.shape[0]):
        X_result[i, ...] = X[order[i], ...]
        y_result[i, ...] = y[order[i], ...]
    return X_result, y_result



if __name__ == "__main__":
    t = 0

    #新規
    
    t,v,tD,vD ,s= generate_coch(save_arr=1,save=0)
    
    print(t.shape,v.shape,tD.shape,vD.shape)
    #作成済み
    t2,v2,tD2,vD2 ,s2= load_datasets()
    t3 = t2.reshape((250,50,86))
    t4 = t3.reshape((250*50,86))
    #print(t4==t2)
    for i in range(250):
         print(vD2[i][0])
    #if t and t2:print(t==t2)