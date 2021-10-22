import argparse
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn import datasets
# import keras


# def generate_image(num=1000):
#     X, y = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)

#     X_train, X_test, y_train, y_test = train_test_split(X / 255, # ピクセル値が 0 - 1 になるようにする
#                                                             y.astype('int64'), # 正解データを数値にする
#                                                             stratify = y,
#                                                             random_state=1)
#     y_train   = keras.utils.to_categorical(y_train, 10)
#     y_test   = keras.utils.to_categorical(y_test, 10)
#     return X_train[:num], X_test[:num], y_train[:num], y_test[:num]

def load_datasets():
    cwd = "./image_matrix_dir/"

    x_train = np.load(cwd + "image500_x_train.npy")
    x_test  = np.load(cwd + "image500_x_test.npy")
    y_train = np.load(cwd + "image500_y_train.npy")
    y_test  = np.load(cwd + "image500_y_test.npy")

    return  x_train, x_test, y_train, y_test

if __name__ == "__main__":   
    x_train,x_test,y_train,y_test = generate_image(num=500)
    print(np.sum(y_train,axis = 0))

    if 1:
        np.save("./image_matrix_dir/image500_x_train.npy",arr=x_train)
        np.save("./image_matrix_dir/image500_x_test.npy",arr=x_test)
        np.save("./image_matrix_dir/image500_y_train.npy",arr=y_train)
        np.save("./image_matrix_dir/image500_y_test.npy",arr=y_test)
