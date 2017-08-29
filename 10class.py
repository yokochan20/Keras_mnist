import numpy as np
import keras

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input
from keras import optimizers

x = np.load("/Users/SatoshiYokoyama/D-HACKS/image.npy")
y = np.load("/Users/SatoshiYokoyama/D-HACKS/labels.npy")

#one_hot vector
y = label_binarize(y,classes=list(range(10)))
#train:test=6:1
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=1./7)
#train:valid=11:1
x_train, x_valid, y_train, y_valid = train_test_split(x_train,y_train,test_size=1./12)
#モデル構築

#１種類の入力，１種類の出力だと簡単
model = Sequential()
model.add(Dense(500,input_shape=(784,)))
model.add(Activation("relu"))
model.add(Dense(10))
model.add(Activation("softmax"))
'''
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
image = Input(shape=(784,))
h = Dense(500)(image)
h = Activation("relu")(h)
h = Dense(10)(h)
y = Activation("softmax")(h)
model = Model(inputs=image,outputs=y)
'''


#学習
model.compile(optimizer="rmsprop",
      loss="categorical_crossentropy",
      metrics=["accuracy"])

model.fit(x_train,y_train,batch_size=32,epochs=10)

#評価
model.evaluate(x_test,y_test,metrics=["accuracy"])
