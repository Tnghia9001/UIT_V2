"""Lite R-ASPP Semantic Segmentation based on MobileNetV3.
"""

import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Conv2D, AveragePooling2D, BatchNormalization, Activation, Multiply, Add, Reshape, Lambda, ReLU, Dropout, Flatten, Dense, Input, Concatenate, Embedding
from keras.utils.vis_utils import plot_model
from model.layers.bilinear_upsampling import BilinearUpSampling2D
# from tensorflow.image import ResizeMethod

class Angle_model_v3:
    def __init__(self, input_shape, n_class=1, alpha=1.0, weights=None, backbone='small'):
        """Init.

        # Arguments
            input_shape: An integer or tuple/list of 3 integers, shape
                of input tensor (should be 1024 × 2048 or 512 × 1024 according
                to the paper).
            n_class: Integer, number of classes.
            alpha: Integer, width multiplier for mobilenetV3.
            weights: String, weights for mobilenetv3.
            backbone: String, name of backbone (must be small or large).
        """
        self.shape = input_shape
        self.n_class = n_class
        self.alpha = alpha
        self.weights = weights
        self.backbone = backbone

    def _extract_backbone(self):
        """extract feature map from backbone.
        """
        if self.backbone == 'small':
            from model.mobilenet_v3_small import MobileNetV3_Small

            model = MobileNetV3_Small(self.shape, self.n_class, alpha=self.alpha, include_top=False).build()
            layer_name8 = 'batch_normalization_7'
            layer_name16 = 'add_2'
        else:
            raise Exception('Invalid backbone: {}'.format(self.backbone))

        if self.weights is not None:
            model.load_weights(self.weights, by_name=True)

        inputs= model.input
        # 1/8 feature map.
        out_feature8 = model.get_layer(layer_name8).output
        # 1/16 feature map.
        out_feature16 = model.get_layer(layer_name16).output

        return inputs, out_feature8, out_feature16

    def build(self, plot=False):
        """build Lite R-ASPP.

        # Arguments
            plot: Boolean, weather to plot model.

        # Returns
            model: Model, model.
        """
        inputs, out_feature8, out_feature16 = self._extract_backbone()

        input1 = Input(shape=(1,))
        # input2 = Lambda(lambda input2: (input2/50))(input1)

        x5 = Embedding(3, 8)(input1)
        x5 = Dense(16)(x5)
        x5 = Reshape((1, 1, 16))(x5)

        # branch1
        x1 = Conv2D(128, (1, 1))(out_feature16)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)

        # branch2
        s = x1.shape
        x2 = AveragePooling2D(pool_size=(int(s[1]), int(s[2])), strides=(16, 20))(out_feature16)

        x2 = Conv2D(128, (1, 1))(x2)
        x2 = Activation('sigmoid')(x2)

        x4 = Dropout(0.2)(x2)
        x4 = Concatenate()([x5, x4])
        x4 = Flatten()(x4)

        x4 = Dense(100, activation='elu')(x4)
        x4 = Dropout(0.2)(x4)
        x4 = Dense(50, activation='elu')(x4)
        x4 = Dense(10, activation='elu')(x4)
        x = Dense(1)(x4)

        model = Model(inputs=[inputs, input1], outputs=x)

        if plot:
            plot_model(model, to_file='LR_ASPP.png', show_shapes=True)

        return model
#
# model = Angle_model_v3((64, 128,3)).build()
# # model.summary()
# #
# # model.load_weights("model-203.h5")
# import cv2
# import numpy as np
# img = cv2.imread('28102020_1.jpg')
# img = cv2.resize(img[100:,:,:], (128,64),cv2.INTER_AREA)
# img = np.array([img])
# print(img.shape)
# bbs = np.array([0])
# print (bbs.shape)
# ang = model.predict([img,bbs],batch_size=1)[0]
# print(ang)

