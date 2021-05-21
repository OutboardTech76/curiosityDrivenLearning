"""
Main module of the Curiosity Driven Learning Project
"""
import retro
from utils import action_helper as ah
import tensorflow as tf
import numpy as np
import ipdb

DEBUG = False

ModelCNN = tf.keras.Model

### ACTION SET
### X   X   X   X   X   X   X   X   X
### B   0   s   S   U   D   L   R   A
# 0 = NULL
# s = select
# S = start

MARIO_ACTION_MASK =[1,0,0,0,0,1,1,1,1]


class InvModel(tf.keras.Model):
    """
    Class to create model for inverse module.
    Creates model declared in paper:
    4 cnn layers with 32 filters each,
    kernel 3x3, stride 2 and padding 1.
    Activation ELU.
    """

    def __init__(self, img_shape, n_outputs):
        super(InvModel, self).__init__()
        kernel = (3, 3)
        padding = 1
        stride = 2
        nFilter = 32
        inp_layer = tf.keras.Input(shape = img_shape)

        self.conv1 = tf.keras.layers.Conv2D(nFilter, kernel, strides=stride, padding='same', activation='elu')
        self.conv2 = tf.keras.layers.Conv2D(nFilter, kernel, strides=stride, padding='same', activation='elu')
        self.conv3 = tf.keras.layers.Conv2D(nFilter, kernel, strides=stride, padding='same', activation='elu')
        self.conv4 = tf.keras.layers.Conv2D(nFilter, kernel, strides=stride, padding='same', activation='elu')

        self.flat1 = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(256,activation='elu')
        self.dense2 = tf.keras.layers.Dense(n_outputs,activation='softmax')

        self._outputs = None

        self.out = self.call(inp_layer)
        super().__init__(inputs = inp_layer, outputs = self.out)

    def call(self, inputs, training=None, mask=None):
        # print("inputs shape: {}".format(inputs.shape))
        # print("batch shape: {}".format(self.conv1._batch_input_shape))
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flat1(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x
        




if __name__=="__main__":

    labels = []
    env = retro.make(game="SuperMarioBros-Nes")
    obs = env.reset()
    print("Image shape: {}".format(obs.shape))
    model = InvModel(obs.shape, sum(MARIO_ACTION_MASK))
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    count = 0
    Xs = []
    while True:
        action_taken = env.action_space.sample()
        # print(ah.full_action_2_partial_action(action_taken,MARIO_ACTION_MASK))
        obs, rew, done, info = env.step(action_taken)
        env.render()
        # test = obs.reshape(1, 224, 240, 3)
        # test = obs.reshape((1,) + obs.shape)
        obs = np.array(obs,dtype=np.float32)
        obs = obs / 255.0
        Xs.append(obs)
        # print("Shape input: {}, Input type: {}".format(test.shape, type(test)))
        count += 1


        if DEBUG:
            break
        if done:
            obs = env.reset()
            break
    env.close()
    m_length = len(Xs)
    for _ in range(m_length):
        vec = np.random.random_sample(5)
        vec = [np.around(i,decimals=0) for i in vec]
        labels.append(vec)
    # ipdb.set_trace()
    Xs = np.array(Xs,np.float32)
    labels = np.array(labels)
    print("Count: {}".format(count))
    print("Shape: ({},{},{},{})".format(len(Xs),Xs[0].shape[0],Xs[0].shape[1],Xs[0].shape[2]))
    model.fit(Xs,labels,epochs=20, batch_size=8)
