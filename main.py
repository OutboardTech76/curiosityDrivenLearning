"""
Main module of the Curiosity Driven Learning Project
"""
import retro
from utils import action_helper as ah
import tensorflow as tf
import numpy as np

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

    def __init__(self):
        super(InvModel, self).__init__()
        kernel = (3, 3)
        padding = 1
        stride = 2
        nFilter = 32
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), strides=(1,1), padding='same', activation='relu', input_shape=(224,240,3))
        self.conv2 = tf.keras.layers.Conv2D(32, (3, 3), strides=(1,1), padding='same', activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(32, (3, 3), strides=(1,1), padding='same', activation='relu')
        self.conv4 = tf.keras.layers.Conv2D(32, (3, 3), strides=(1,1), padding='same', activation='relu')
        # self.conv1 = tf.keras.layers.Conv2D(nFilter, kernel, strides=stride, padding='same', activation='elu', input_shape=(224, 240, 3))
        # self.conv2 = tf.keras.layers.Conv2D(nFilter, kernel, strides=stride, padding='same', activation='elu')
        # self.conv3 = tf.keras.layers.Conv2D(nFilter, kernel, strides=stride, padding='same', activation='elu')
        # self.conv4 = tf.keras.layers.Conv2D(nFilter, kernel, strides=stride, padding='same', activation='elu')

    def call(self, inputs):
        x = self.conv1(inputs)
        return self.conv4(inputs)
        




if __name__=="__main__":

    env = retro.make(game="SuperMarioBros-Nes")
    obs = env.reset()
    print("Image shape: {}".format(obs.shape))
    model = InvModel()
    while True:
        action_taken = env.action_space.sample()
        # print(ah.full_action_2_partial_action(action_taken,MARIO_ACTION_MASK))
        obs, rew, done, info = env.step(action_taken)
        env.render()
        test = obs[..., np.newaxis]
        

        model.call(test)
        model.summary()

        if done:
            obs = env.reset()
            break
    env.close()
