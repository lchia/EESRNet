import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, Input, ReLU, Lambda, Add, LayerNormalization, 
                                     Dropout, Permute, Reshape, DepthwiseConv2D, Layer, 
                                     GlobalAveragePooling2D, Dot, Dense, Multiply,Resizing,UpSampling2D)
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_normal 
import tensorflow.keras.backend as K
import numpy as np


def single(scale=3, in_channels=3, num_fea=16, m=5, out_channels=3, netname='xlsrbase7', stage='train'):

    inp = Input(shape=(None, None, in_channels))

    upsample_func = Lambda(lambda x_list: tf.concat(x_list, axis=3))
    upsampled_inp = upsample_func([inp]*(scale**2))

    concat_c_func = Lambda(lambda x_list: tf.concat(x_list, axis=-1))

    if netname == 'base7':

        # Feature extraction
        x = Conv2D(num_fea, 3, padding='same', activation='relu', kernel_initializer=glorot_normal(), bias_initializer='zeros')(inp)

        for i in range(m):
            x = Conv2D(num_fea, 3, padding='same', activation='relu', kernel_initializer=glorot_normal(), bias_initializer='zeros')(x)

        # Pixel-Shufflesd
        x = Conv2D(out_channels*(scale**2), 3, padding='same', activation='relu', kernel_initializer=glorot_normal(), bias_initializer='zeros')(x)
        x = Conv2D(out_channels*(scale**2), 3, padding='same', kernel_initializer=glorot_normal(), bias_initializer='zeros')(x)
        x = Add()([upsampled_inp, x])
        
        depth_to_space = Lambda(lambda x: tf.nn.depth_to_space(x, scale))
        out = depth_to_space(x)
 
    elif netname == 'eesrnet':

        # Feature extraction
        x = Conv2D(num_fea, 3, padding='same', activation='relu', kernel_initializer=glorot_normal(), bias_initializer='zeros')(inp)

        for i in range(m):
            x = Conv2D(num_fea, 3, padding='same', activation='relu', kernel_initializer=glorot_normal(), bias_initializer='zeros')(x)

        # Pixel-Shufflesd
        x = Conv2D(out_channels*(scale**2), 3, padding='same', activation='relu', kernel_initializer=glorot_normal(), bias_initializer='zeros')(x)
        x = Conv2D(out_channels*(scale**2), 3, padding='same', kernel_initializer=glorot_normal(), bias_initializer='zeros')(x)
        #x = Add()([upsampled_inp, x])
        
        depth_to_space = Lambda(lambda x: tf.nn.depth_to_space(x, scale))
        out = depth_to_space(x)
 
    else:
        print(f'\n')
        print(f'> select netname from base7, resbase7, xlsrbase7, ...')
        print(f'\n')
        out = upsampled_inp
 
    clip_func = Lambda(lambda x: K.clip(x, 0., 255.))
    out = clip_func(out) 

    return Model(inputs=inp, outputs=out)

if __name__ == '__main__':
    model = single()
    print('Params: [{:.2f}]K'.format(model.count_params()/1e3))
