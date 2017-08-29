import numpy as np
import keras

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, Conv2D
from keras.layers import Maxpooling2D
from keras import optimizers

x = np.load("/Users/SatoshiYokoyama/D-HACKS/image.npy")
y = np.load("/Users/SatoshiYokoyama/D-HACKS/labels.npy")


def classify_labels():
    y_new = []
    count = 0
    #偶数と奇数に分ける
    #偶数は０，奇数は１
    def classify():
        global y
        for num in y:
            if num % 2 == 0:
                y = 0
            else:
                y = 1
            y_new.append(y)
    classify()
    #one_hot vector
    y_new = label_binarize(y_even,classes=list(range(2)))


#モデル
#１種類の入力，１種類の出力だと簡単
def model1():
    model = Sequential()

    model.add(Conv2D(500,input_shape=(32,32,1)))
    model.add(Maxpooling2D(pool_size=(2,2)))

    model.add(Conv2D(500,input_shape=(32,32,1)))
    model.add(Maxpooling2D(pool_size=(2,2)))

    model.add(Conv2D(500,input_shape=(32,32,1)))
    model.add(Maxpooling2D(pool_size=(2,2)))

    model.add(Dense(2))

def model2():
    model = Sequential([
        Dense(500,input_shape=(784,)),
        Activation("relu"),
        Dense(10),
        Activation("softmax")
        ])

    model = Sequential()
    model.add(Dense(500,input_shape=(784,),activation="relu"))
    model.add(Dense(10,activation="softmax"))

#複数の入力の時はこっち
def model3():
    image = Input(shape=(784,))
    h = Dense(500)(image)
    h = Activation("relu")(h)
    h = Dense(10)(h)
    y = Activation("softmax")(h)
    model = Model(inputs=image,outputs=y)


#学習
#rmspropの理由:
def compile1():
    model.compile(optimizer="rmsprop",
          loss="categorical_crossentropy",
          metrics=["accuracy"])

    model.fit(x_train,y_train,batch_size=32,epochs=10)

#評価
def evaluate1():
    model.evaluate(x_test,y_test,metrics=["accuracy"])


classify_labels()
model1()
compile1()
evaluate1()
