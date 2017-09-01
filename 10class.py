#-*- coding: utf-8 -*-

def import_model():
    import numpy as np
    import keras

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import label_binarize
    from keras.models import Sequential, Model
    from keras.layers import Dense, Activation, Input
    from keras import optimizers
    from sklearn.preprocessing import label_binarize

#def import_dataset():
    x = np.load("/Users/SatoshiYokoyama/D-HACKS/image.npy")
    #y = np.load("/Users/SatoshiYokoyama/D-HACKS/labels.npy")
    y0 = np.load("/Users/SatoshiYokoyama/D-HACKS/labels.npy")

#ラベルをone_hot vectorにする
def one_hot():
    from sklearn.preprocessing import label_binarize
    #global y
    y = label_binarize(y0,classes=list(range(10)))

#train_test_valid_datasetを作成
def mk_dataset():
    #train:test=6:1
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=1./7)
    #train:valid=11:1
    x_train, x_valid, y_train, y_valid = train_test_split(x_train,y_train,test_size=1./12)

#モデル構築
def model1():
    #１種類の入力，１種類の出力だと簡単
    model = Sequential()
    model.add(Dense(500,input_shape=(784,)))
    model.add(Activation("relu"))
    model.add(Dense(10))
    model.add(Activation("softmax"))

def model2():
    model = Sequential([
        Dense(500,input_shape=(784,)),
        Activation("relu"),
        Dense(10),
        Activation("softmax")
        ])

def model3():
    model = Sequential()
    model.add(Dense(500,input_shape=(784,),activation="relu"))
    model.add(Dense(10,activation="softmax"))

def model4():
    #複数の入力の時はこっち
    image = Input(shape=(784,))
    h = Dense(500)(image)
    h = Activation("relu")(h)
    h = Dense(10)(h)
    y = Activation("softmax")(h)
    model = Model(inputs=image,outputs=y)

def train1():
    #学習
    model.compile(optimizer="rmsprop",
          loss="categorical_crossentropy",
          metrics=["accuracy"])

    model.fit(x_train,y_train,batch_size=32,epochs=10)

def evaluate1():
    #評価
    model.evaluate(x_test,y_test,metrics=["accuracy"])

import_model()
#import_dataset()
one_hot()
mk_dataset()
model1()
train1()
evaluate1()
