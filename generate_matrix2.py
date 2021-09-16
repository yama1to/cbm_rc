import numpy as np 
import networkx as nx

    

def generate_wr(seed,N_x, density, rho):
    np.random.seed(seed=seed)   
    # Erdos-Renyiランダムグラフ
    m = int(N_x*(N_x-1)*density/2)  # 総結合数
    G = nx.gnm_random_graph(N_x, m,seed)

    # 行列への変換(結合構造のみ）
    connection = nx.to_numpy_matrix(G)
    W = np.array(connection)

    # 非ゼロ要素を一様分布に従う乱数として生成
    rec_scale = 1.0
    np.random.seed(seed=seed)
    W *= np.random.uniform(-rec_scale, rec_scale, (N_x, N_x))

    # スペクトル半径の計算
    eigv_list = np.linalg.eig(W)[0]
    sp_radius = np.max(np.abs(eigv_list))

    # 指定のスペクトル半径rhoに合わせてスケーリング
    W *= rho / sp_radius

    return W


def generate_win(seed,input_scale,N_x,N_u):
    np.random.seed(seed=seed)

    Win = np.random.uniform(-input_scale, input_scale, (N_x, N_u))

    return Win

def generate_Wout(seed,N_x,N_y):
    np.random.seed(seed=seed)
    Wout = np.random.normal(size=(N_y, N_x))


def generate_mat(seed,N_u,N_x,N_y,density,rho,input_scale,):
    Win = generate_win(seed,input_scale,N_x,N_u)
    Wrec = generate_wr(seed,N_x, density, rho)
    Wout = generate_Wout(seed,N_x,N_y)

    return Win,Wrec,Wout 
