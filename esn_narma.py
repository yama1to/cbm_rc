import numpy as np
import matplotlib.pyplot as plt
from esn_model_text import ESN, Tikhonov

from  generate_data_sequence_narma  import *

if __name__ == '__main__':


    rm = 100
    MM1 = 900
    MM2 = 100
    U,D  = generate_narma(N=MM1+MM2+rm,seed=1)
    U = U[rm:]
    D = D[rm:]
    train_U = U[:MM1]
    test_U = U[MM1:]
    train_D = D[:MM1]
    test_D = D[MM1:]


    # ESNモデル
    N_x = 100  # リザバーのノード数
    model = ESN(train_U.shape[1], train_D.shape[1], N_x, 
                density=0.15, input_scale=0.1, rho=0.9,
                fb_scale=0.1, fb_seed=0)

    # 学習（リッジ回帰）
    train_Y = model.train(train_U, train_D, 
                          Tikhonov(N_x, train_D.shape[1], 1e-4)) 

    # モデル出力
    test_Y = model.predict(test_U)

    # 評価（テスト誤差RMSE, NRMSE）
    RMSE = np.sqrt(((test_D - test_Y) ** 2).mean())
    NRMSE = RMSE/np.sqrt(np.var(test_D))
    print('RMSE =', RMSE)
    print('NRMSE =', NRMSE)

    plt.plot(test_Y)
    plt.plot(test_D)
    plt.show()

    # グラフ表示用データ
    T_disp = (-100, 100)
    t_axis = np.arange(T_disp[0], T_disp[1])
    disp_U = np.concatenate((train_U[T_disp[0]:], test_U[:T_disp[1]]))
    disp_D = np.concatenate((train_D[T_disp[0]:], test_D[:T_disp[1]]))
    disp_Y = np.concatenate((train_Y[T_disp[0]:], test_Y[:T_disp[1]]))

    # グラフ表示
    plt.rcParams['font.size'] = 12
    fig = plt.figure(figsize=(7, 5))
    plt.subplots_adjust(hspace=0.3)

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.text(-0.15, 1, '(a)', transform=ax1.transAxes)
    ax1.text(0.2, 1.05, 'Training', transform=ax1.transAxes)
    ax1.text(0.7, 1.05, 'Testing', transform=ax1.transAxes)
    plt.plot(t_axis, disp_U[:,0], color='k')
    plt.ylabel('Input')
    plt.axvline(x=0, ymin=0, ymax=1, color='k', linestyle=':')

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.text(-0.15, 1, '(b)', transform=ax2.transAxes)
    plt.plot(t_axis, disp_D[:,0], color='k', label='Target')
    plt.plot(t_axis, disp_Y[:,0], color='gray', linestyle='--', label='Model')
    plt.xlabel('n')
    plt.ylabel('Output')
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right')
    plt.axvline(x=0, ymin=0, ymax=1, color='k', linestyle=':')

    plt.show()
