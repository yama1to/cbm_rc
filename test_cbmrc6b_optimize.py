# Copyright (c) 2018 Katori lab. All Rights Reserved
# NOTE:

import optimization as opt
opt.parallel=40
opt.exe="python cbmrc6b.py  ex=opt1 display=0"
opt.filename_tmp="data_cbmrc6b_opt1.csv"
opt.filename_rec="data_cbmrc6b_opt1_rec.csv"

opt.clear()
opt.appendid()
opt.appendseed("seed")# 乱数のシード
opt.appendxfr("alpha_r",0.2,0,1,2)
opt.appendxfr("alpha_b",0.2,0,1,2)
opt.appendxfr("alpha_s",0.2,0,2,2)
opt.config()
opt.names=['seed','id','NN','Nh','alpha_i','alpha_r','alpha_b','alpha_s','alpha0','alpha1','beta_i','beta_r','beta_b','Temp','lambda0','RMSE1','RMSE2','count_gap']

def func(row):
    return row['RMSE1'] + 0.1*row['count_gap']

#opt.minimize({'target':"y1",'iteration':3,'population':10,'samples':5})
opt.minimize({'TARGET':func,'iteration':10,'population':20,'samples':20})
