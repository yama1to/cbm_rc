# Copyright (c) 2018-2020 Katori lab. All Rights Reserved
# NOTE:
"""
explorerのテスト用プログラム（変数に文字列を含む場合）
"""
import pandas as pd
import pickle
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from explorer import common

def rel(x):
    return np.fmax(x,0)
def tanh(x):
    return np.tanh(x)
def sin(x):
    return np.sin(x)

func_dic = {"rel": rel, "tanh": tanh, "sin":sin}

class Config():
    def __init__(self):
        # columns, csv, id: データの管理のために必須の変数
        self.columns = None # 結果をCSVに保存する際のコラム
        self.csv = None # 結果を保存するファイル
        self.id  = None
        self.plot = True # matplotlibによる出力のオンオフ
        self.show = True # 図の表示のオンオフ、explorerは実行時にこれをオフにする。
        self.fig1 = "fig1.png" ### 画像ファイル名

        # config
        self.id = 1 # パラメータ値のID。モデルの動作には関わらないが、データの管理のために必須。
        self.seed=0 # 乱数生成のためのシード
        self.x1 = 2.0
        self.x2 = 1.0
        self.x3 = 0.5

        self.f1 = "rel"

        # output
        self.y1 = None
        self.y2 = None
        self.y3 = None
        self.y4 = None
        ##
        self.fig1 = "fig1.png" ### 画像ファイル名
        self.plot = False # matplotlibによる出力のオンオフ

def execute():
    #time.sleep(5)
    f1=func_dic[c.f1]
    np.random.seed(int(c.seed))
    c.y1 = c.x1*c.x1 + c.x2*c.x2 + c.x3*c.x3 + np.random.normal(0.0,0.1)
    c.y2 = 10*2 + c.x1*c.x1-10*np.cos(2*np.pi*c.x1) + c.x2*c.x2-10*np.cos(1*np.pi*c.x2*c.x3) + np.random.normal(0.0,10)
    c.y3 = np.sin(c.x1)**10 + np.cos(10 + c.x2*c.x1) * np.cos(c.x1)
    c.y4 = f1(c.x1)

def plot(c):
    x=np.arange(3)
    y=(c.y1,c.y2,c.y3)
    plt.figure()
    plt.bar(x,y)
    plt.ylabel("y")
    plt.xlabel("x")
    plt.savefig(c.fig1) ### 画像の保存
    if c.show: plt.show()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-config", type=str)
    a = ap.parse_args()

    c=Config()
    if a.config: c=common.load_config(a)
    execute()
    print("c.x1",c.x1)
    if a.config: common.save_config(c)
    if c.plot: plot(c)
