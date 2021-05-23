"""
Main module of the Curiosity Driven Learning Project
"""
import retro
from utils import action_helper as ah
import tensorflow as tf
import numpy as np
from nptyping import NDArray
import ipdb

# DEBUG = False
DEBUG = True

MISSING = object()

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

    Args:
        obs: Image got from the environment
        n_outputs: Amount of actions that is able to take
    """

    def __init__(self, obs: NDArray, n_outputs: int, kernel= (3,3), padding= 'same', stride= 2, nFilters= 32):
        super(InvModel, self).__init__()

        inp_layer = tf.keras.Input(shape = obs.shape)

        self.conv1 = tf.keras.layers.Conv2D(nFilters, kernel, strides=stride, padding=padding, activation='elu')
        self.conv2 = tf.keras.layers.Conv2D(nFilters, kernel, strides=stride, padding=padding, activation='elu')
        self.conv3 = tf.keras.layers.Conv2D(nFilters, kernel, strides=stride, padding=padding, activation='elu')
        self.conv4 = tf.keras.layers.Conv2D(nFilters, kernel, strides=stride, padding=padding, activation='elu')

        self.flat1 = tf.keras.layers.Flatten()

        self.concat = tf.keras.layers.Concatenate(axis=1)

        self.dense1 = tf.keras.layers.Dense(256,activation='elu')
        self.dense2 = tf.keras.layers.Dense(n_outputs,activation='softmax')

        self.features_out = self.flat1(self.conv4(self.conv3(self.conv2(self.conv1(inp_layer)))))
        self.features = tf.keras.Model(inputs=[inp_layer], outputs=[self.features_out], name='Inverse Model')

        state = self.features(obs.reshape((1,)+obs.shape))
        self.prevState = tf.convert_to_tensor(state, dtype=tf.float32)

        self.out = self.call(inp_layer)
        super(InvModel, self).__init__(inputs = inp_layer, outputs = self.out)

    def call(self, inputs, training=None, mask=None):
        """
        Function for passing an element through the model

        Args:
            inputs: the input tensor that we will be forwarding through
            training: Flag that will be passed when training, unused
            mask: Unused

        Returns:
            Tensor result of forwarding the input through the net
        """
        prev_state = self.prevState
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flat1(x)       # Features output
        curr_state = x
        x = self.concat([prev_state, curr_state])
        x = self.dense1(x)
        x = self.dense2(x)
        self.auxPrevState = curr_state
        self.auxPrevState = self.prevState
        return x

    def encode_state(self, image: NDArray):
        """
        Function for converting the image got from the environment to 
        its feature-representation

        Args:
            image: Image obtained from the environment

        Returns:
            Feature-state representation of the image given
        """
        in_image =  image.reshape((1, ) + image.shape)
        return self.features(in_image)


class FwdModel(tf.keras.Model):
    """
    Class to create model for forward module.
    Creates model declared in paper:
    2 Fully connected layers

    Args:
        feature_dim: Size of a feature vector
        action_dim: Size of an action vector
    """
    def __init__(self, feature_dim: int, action_dim: int, middle_layer_size= 256):
        super(FwdModel, self).__init__()

        inp_layer = tf.keras.Input(shape = feature_dim+action_dim)

        self.dense1 = tf.keras.layers.Dense(middle_layer_size)
        self.dense2 = tf.keras.layers.Dense(feature_dim)

        self.out = self.call(inp_layer)
        super(FwdModel,self).__init__(inputs = inp_layer, outputs = self.out)

    def fromInvModel(iModel: InvModel):
        """
        Function for constructing a FwdModel from an InvModel
        
        Args:
            iModel: The InvModel that this FwdModel will be paired to

        Returns:
            A constructed FwdModel that is compatible with the given InvModel
        """
        action_dim = iModel.dense2.output_shape[1]
        features_dim = iModel.flat1.output_shape[1]
        return FwdModel(features_dim, action_dim)

    def call(self, inputs, training=None, mask=None):
        """
        Function for passing an element through the model

        Args:
            inputs: the input tensor that we will be forwarding through
            training: Flag that will be passed when training, unused
            mask: Unused

        Returns:
            Tensor result of forwarding the input through the net
        """
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x


def L2Norm(y_true, y_pred):
    """
    Function for calculating the L2 Norm between the label and the predicted values

    Args:
        y_true: Label
        y_pred: Prediction

    Returns:
        L2 Norm between y_pred and y_true
    """
    return tf.keras.backend.mean(y_true-y_pred)**2

class IntrinsicCuriosityModule():
    def __init__(self, obs: NDArray, action_dim: int, verbose: bool = False):
        self.iModel = InvModel(obs,action_dim)
        self.iModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
        self.fModel = FwdModel.fromInvModel(self.iModel)
        self.fModel.compile(optimizer='adam', loss=L2Norm)
        if verbose:
            print("InvModel summary:")
            self.iModel.summary()
            print("FwdModel summary:")
            self.fModel.summary()
        
        

if __name__=="__main__":

    if DEBUG:
        count = 0
    env = retro.make(game="SuperMarioBros-Nes")
    obs = env.reset()
    icm = IntrinsicCuriosityModule(obs,sum(MARIO_ACTION_MASK))
    prevStatePlusAct = None
    while True:
        action_taken = env.action_space.sample()
        action_rep = ah.full_action_2_partial_action(action_taken,MARIO_ACTION_MASK)
        action_lbl = np.array(action_rep)
        obs, rew, done, info = env.step(action_taken)

        # env.render()

        obs = np.array(obs,dtype=np.float32)
        obs = obs / 255.0
        in_obs = obs.reshape((1,)+obs.shape)
        in_action_lbl = action_lbl.reshape((1,)+action_lbl.shape)

        state_feature_rep = icm.iModel.encode_state(obs)

        print("InvModel fit")
        icm.iModel.fit(in_obs,in_action_lbl)

        if not prevStatePlusAct is None:
            print("FwdModel fit")
            icm.fModel.fit(prevStatePlusAct,state_feature_rep)

        prevStatePlusAct = tf.concat([state_feature_rep, in_action_lbl], axis=1)

        if DEBUG:
            count += 1
            if count > 10:
                break
        if done:
            obs = env.reset()
            break
    env.close()
