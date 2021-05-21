"""
Main module of the Curiosity Driven Learning Project
"""
import retro
from utils import action_helper as ah
import tensorflow as tf
import numpy as np
import ipdb

DEBUG = False
DEBUG = True

ModelCNN = tf.keras.Model

### ACTION SET
### X   X   X   X   X   X   X   X   X
### B   0   s   S   U   D   L   R   A
# 0 = NULL
# s = select
# S = start

MARIO_ACTION_MASK =[1,0,0,0,0,1,1,1,1]

class FwdModel(tf.keras.Model):
    def __init__(self):
        super(FwdModel, self).__init__()

        inp_layer = tf.keras.Input(shape = 5)

        self.dense1 = tf.keras.layers.Dense(256)
        self.dense2 = tf.keras.layers.Dense(288)

        self.out = self.call(inp_layer)

    def call(self, inputs, training=None, mask=None):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x

class InvModel(tf.keras.Model):
    """
    Class to create model for inverse module.
    Creates model declared in paper:
    4 cnn layers with 32 filters each,
    kernel 3x3, stride 2 and padding 1.
    Activation ELU.
    """

    def __init__(self, obs, n_outputs):
        super(InvModel, self).__init__()
        kernel = (3, 3)
        padding = 1
        stride = 2
        nFilter = 32
        inp_layer = tf.keras.Input(shape = obs.shape)


        self.conv1 = tf.keras.layers.Conv2D(nFilter, kernel, strides=stride, padding='same', activation='elu')
        self.conv2 = tf.keras.layers.Conv2D(nFilter, kernel, strides=stride, padding='same', activation='elu')
        self.conv3 = tf.keras.layers.Conv2D(nFilter, kernel, strides=stride, padding='same', activation='elu')
        self.conv4 = tf.keras.layers.Conv2D(nFilter, kernel, strides=stride, padding='same', activation='elu')

        self.flat1 = tf.keras.layers.Flatten()

        self.concat = tf.keras.layers.Concatenate(axis=1)

        self.dense1 = tf.keras.layers.Dense(256,activation='elu')
        self.dense2 = tf.keras.layers.Dense(n_outputs,activation='softmax')

        self.features_out = self.flat1(self.conv4(self.conv3(self.conv2(self.conv1(inp_layer)))))
        self.features = tf.keras.Model(inputs=[inp_layer], outputs=[self.features_out], name='Inverse Model')

        self.prev_state = self.features(obs.reshape((1,)+obs.shape))

        self.out = self.call(inp_layer)
        super(InvModel, self).__init__(inputs = inp_layer, outputs = self.out)

    def call(self, inputs, training=None, mask=None):
        # print("inputs shape: {}".format(inputs.shape))
        # print("batch shape: {}".format(self.conv1._batch_input_shape))
        prev_state = self.prev_state
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flat1(x)       # Features output
        curr_state = x
        print(inputs.shape)
        x = self.concat([prev_state,curr_state])
        x = self.dense1(x)
        x = self.dense2(x)
        self.prev_state = curr_state
        return x

    # def train_step(self, data):
        # x,y = data

        # with tf.GradientTape() as tape:
            # y_pred = self(x,training=True)
            # loss = self.compiled_loss(y,y_pred, regularization_losses=self.losses)

        # trainable_vars = self.trainable_variables
        # gradients = tape.gradient(loss, trainable_vars)
        # self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # self.compiled_metrics.update_state(y, y_pred)
        # return {m.name: m.result() for m in self.metrics}


    def encode_state(self, image):
        return self.features(image)
        

    


if __name__=="__main__":

    env = retro.make(game="SuperMarioBros-Nes")
    # ipdb.set_trace()
    obs = env.reset()
    print("Image shape: {}".format(obs.shape))
    model = InvModel(obs, sum(MARIO_ACTION_MASK))
    model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    while True:
        action_taken = env.action_space.sample()
        lbl = np.array(ah.full_action_2_partial_action(action_taken,MARIO_ACTION_MASK))
        obs, rew, done, info = env.step(action_taken)

        # env.render()

        obs = np.array(obs,dtype=np.float32)
        obs = obs / 255.0
        
        obs = obs.reshape((1,)+obs.shape)
        lbl = lbl.reshape((1,)+lbl.shape)
        model.fit(obs,lbl)

        if DEBUG:
            break
        if done:
            obs = env.reset()
            break
    env.close()
