import tensorflow.keras as keras
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Lambda, TimeDistributed, LSTM, Bidirectional, Flatten, GRU
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, Bidirectional, SpatialDropout2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization, Flatten
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import Input
from tensorflow.keras.constraints import max_norm
import numpy as np
import tensorflow as tf
from keras import backend as K

#################################################### model our proposed ##########################################################################      
class TransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Dense(embed_dim, activation='elu', use_bias=False)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout = Dropout(rate)

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout(ffn_output)
        out = self.layernorm2(out1 + ffn_output)
        return out

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'att': self.att,
            'ffn': self.ffn,
            'layernorm1': self.layernorm1,
            'layernorm2': self.layernorm2,
            'dropout': self.dropout
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class PositionalEncoding(keras.layers.Layer):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        self.max_len = max_len
        self.d_model = d_model

    def get_angles(self, pos, i):
        angles = 1 / np.power(10000, (2 * (i//2)) / np.float32(self.d_model))
        return pos * angles

    def call(self, inputs):
        # Get the length of the input sequence
        length = inputs.shape[1]
        # Calculate the positional encoding matrix
        pos_encoding = np.zeros((length, self.d_model))
        for pos in range(length):
            for i in range(self.d_model):
                if i % 2 == 0:
                    pos_encoding[pos, i] = np.sin(self.get_angles(pos, i))
                else:
                    pos_encoding[pos, i] = np.cos(self.get_angles(pos, i))
        pos_encoding = pos_encoding[np.newaxis, ...]
        # Add the positional encoding matrix to the input embeddings
        return inputs + tf.cast(pos_encoding, tf.float32)
    def get_config(self):
        config = super().get_config().copy()
        config.update({'max_len': self.max_len, 'd_model': self.d_model})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# ERTNet model
def ertnet(nb_classes, Chans=64, Samples=128,
           dropoutRate=0.5, kernLength=64, F1=8, heads=8, 
           D=2, F2=16):
    """
    ERTNet(Emotion Recognition Transformer Network) model for EEG Emotion Recognition
    nb_classes: number of classes
    Chans: number of channels in the EEG data
    Samples: number of time points in the EEG data
    dropoutRate: dropout fraction
    kernLength: length of temporal convolution in first layer
    F1: number of temporal filters in first layer
    heads: number of heads for multi-head attention
    D: number of depthwise convolutional filters
    F2: number of separable convolutional filters

    return ERTNet model
    """
    input1 = Input(shape=(Chans, Samples, 1))

    block1 = Conv2D(F1, (1, kernLength), padding='same',
                    input_shape=(Chans, Samples, 1),
                    use_bias=False)(input1)
    block1 = BatchNormalization()(block1)

    block1 = DepthwiseConv2D((Chans, 1), use_bias=False,
                             depth_multiplier=D,
                             depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = Dropout(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, (1, 16),
                             use_bias=False, padding='same')(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = Lambda(lambda x: keras.backend.squeeze(x, axis=1))(block2)

    #position_encoding
    pos_en = PositionalEncoding(F2, 500)(block2)

    attention1 = TransformerBlock(embed_dim=F2,num_heads=heads,rate=dropoutRate)

    transformer1 = attention1(pos_en)
    outputs = GlobalAveragePooling1D()(transformer1)
    outputs = Dropout(dropoutRate)(outputs)
    dense = Dense(nb_classes,activation=None, use_bias=False)(outputs)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input1, outputs=softmax)
##################################################################################################################################################  

def DeepConvNet(nb_classes, Chans = 64, Samples = 256,
                dropoutRate = 0.5):
    """ Keras implementation of the Deep Convolutional Network as described in
    Schirrmeister et. al. (2017), Human Brain Mapping.
    """

    # start the model
    input_main   = Input((Chans, Samples, 1))
    block1       = Conv2D(25, (1, 2),
                                 input_shape=(Chans, Samples, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(input_main)
    block1       = Conv2D(25, (Chans, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block1       = BatchNormalization(epsilon=1e-05, momentum=0.1)(block1)
    block1       = Activation('relu')(block1)
    block1       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block1)
    block1       = Dropout(dropoutRate)(block1)
  
    block2       = Conv2D(50, (1, 2),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block2       = BatchNormalization(epsilon=1e-05, momentum=0.1)(block2)
    block2       = Activation('relu')(block2)
    block2       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block2)
    block2       = Dropout(dropoutRate)(block2)
    
    block3       = Conv2D(100, (1, 2),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block2)
    block3       = BatchNormalization(epsilon=1e-05, momentum=0.1)(block3)
    block3       = Activation('relu')(block3)
    block3       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block3)
    block3       = Dropout(dropoutRate)(block3)
    
    block4       = Conv2D(200, (1, 5),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block3)
    block4       = BatchNormalization(epsilon=1e-05, momentum=0.1)(block4)
    block4       = Activation('relu')(block4)
    block4       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block4)
    block4       = Dropout(dropoutRate)(block4)
    
    flatten      = Flatten()(block4)
    
    dense        = Dense(nb_classes, kernel_constraint = max_norm(0.5))(flatten)
    softmax      = Activation('softmax')(dense)
    
    return Model(inputs=input_main, outputs=softmax)


# need these for ShallowConvNet
def square(x):
    return K.square(x)

def log(x):
    return K.log(K.clip(x, min_value = 1e-7, max_value = 10000))   


def ShallowConvNet(nb_classes, Chans = 64, Samples = 128, dropoutRate = 0.5):
    """ Keras implementation of the Shallow Convolutional Network as described
    in Schirrmeister et. al. (2017), Human Brain Mapping.

    Code can be found at:https://github.com/MiChongGET/EEGNet-TensorFlow
    """

    # start the model
    input_main   = Input((Chans, Samples, 1))
    block1       = Conv2D(40, (1, 13),
                        input_shape=(Chans, Samples, 1),
                        kernel_constraint = max_norm(2., axis=(0,1,2)))(input_main)
    block1       = Conv2D(40, (Chans, 1), use_bias=False, 
                          kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block1       = BatchNormalization(epsilon=1e-05, momentum=0.9)(block1)
    block1       = Activation(square)(block1)
    block1       = AveragePooling2D(pool_size=(1, 7), strides=(1, 3))(block1)
    block1       = Activation(log)(block1)
    block1       = Dropout(dropoutRate)(block1)
    flatten      = Flatten()(block1)
    dense        = Dense(nb_classes, kernel_constraint = max_norm(0.5))(flatten)
    softmax      = Activation('softmax')(dense)
    
    return Model(inputs=input_main, outputs=softmax)

def EEGNet(nb_classes, Chans=64, Samples=128,
           dropoutRate=0.5, kernLength=64, F1=8,
           D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    """
    Original EEGNet model.
    Code can be found at:https://github.com/MiChongGET/EEGNet-TensorFlow
    """
    input1 = Input(shape=(Chans, Samples, 1))

    block1 = Conv2D(F1, (1, kernLength), padding='same',
                    input_shape=(Chans, Samples, 1),
                    use_bias=False)(input1)
    block1 = BatchNormalization()(block1)

    block1 = DepthwiseConv2D((Chans, 1), use_bias=False,
                             depth_multiplier=D,
                             depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('relu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = dropoutType(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, (1, 16),
                             use_bias=False, padding='same')(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('relu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = dropoutType(dropoutRate)(block2)

    flatten = Flatten(name='flatten')(block2)

    dense = Dense(nb_classes, name='dense',
                  kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input1, outputs=softmax)

'''
def CNN_BiLSTM(nb_classes, Chans = 64, Samples = 128, dropoutRate = 0.5, num_lstm=64, F1=40, F2=40):

    # CNN+BiLSTM model based ShallowConvNet and BiLSTM
    input_main   = Input((Chans, Samples, 1))
    block1       = Conv2D(F1, (1, 13),
                        input_shape=(Chans, Samples, 1),
                        kernel_constraint = max_norm(2., axis=(0,1,2)))(input_main)
    block1       = Conv2D(F2, (Chans, 1), use_bias=False, 
                          kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block1       = BatchNormalization(epsilon=1e-05, momentum=0.9)(block1)
    block1       = Activation(square)(block1)
    block1       = AveragePooling2D(pool_size=(1, 7), strides=(1, 3))(block1)
    block1       = Activation(log)(block1)
    block1       = Lambda(lambda x: keras.backend.squeeze(x, axis=1))(block1)
    block1       = Dropout(dropoutRate)(block1)
    reshaped = TimeDistributed(Flatten())(block1)

    bilstm = Bidirectional(LSTM(num_lstm, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))(reshaped)

    flatten = Flatten(name='flatten')(bilstm)
    dropout1 = Dropout(dropoutRate)(flatten)

    output_layer = Dense(nb_classes, activation='softmax')(dropout1)
    return Model(inputs=input_main, outputs=output_layer)
'''

def CNN_BiLSTM(nb_classes, Chans=64, Samples=128,
           dropoutRate=0.5, kernLength=64, F1=8, num_lstm=64,
           D=2, F2=16, dropoutType='Dropout'):
    # CNN+BiLSTM model based EEGNet and BiLSTM
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(Chans, Samples, 1))

    block1 = Conv2D(F1, (1, kernLength), padding='same',
                    input_shape=(Chans, Samples, 1),kernel_regularizer=tf.keras.regularizers.l2(0.01),
                    use_bias=False)(input1)
    block1 = BatchNormalization()(block1)

    block1 = DepthwiseConv2D((Chans, 1), use_bias=False,
                             depth_multiplier=D,kernel_regularizer=tf.keras.regularizers.l2(0.01),
                             depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = dropoutType(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, (1, 16),kernel_regularizer=tf.keras.regularizers.l2(0.01),
                             use_bias=False, padding='same')(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    reshaped = TimeDistributed(Flatten())(block1)

    bilstm = GRU(num_lstm)(reshaped)

    dropout1 = Dropout(dropoutRate)(bilstm)
    output_layer = Dense(nb_classes, activation='softmax')(dropout1)
    return Model(inputs=input1, outputs=output_layer)

  
def gru_net(nb_classes, Chans=32, Samples=512, dropoutRate=0.3, L1=32, L2=16):

    # Pure GRU model
    input_layer = Input(shape=(Chans, Samples, 1))  
    timedist = TimeDistributed(Flatten())(input_layer)  
    bigru = Bidirectional(GRU(L1, return_sequences=True, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))(timedist)
    dropout1 = Dropout(dropoutRate)(bigru)
    gru = GRU(L2, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(dropout1)
    output_layer = Dense(nb_classes, activation='softmax')(gru)

    return Model(inputs=input_layer, outputs=output_layer)
