import tensorflow as tf
import numpy as np
import random

width_mult_list = [1/32, 0.125, 0.25, 0.50, 1.0]
channels = [int(64 * width_mult) for width_mult in width_mult_list]
start_C_1 = [7/8, 3/4,1/2,0,0]

class WeightSharingConv2d(tf.keras.layers.Conv2D):
    def __init__(self, in_channels_list, out_channels_list,
                 kernel_size, stride=1, padding=0, dilation=1,
                 groups_list=[1], bias=True):
        
        super(WeightSharingConv2d, self).__init__(
            max(out_channels_list), (kernel_size, kernel_size),
            strides=(stride, stride), padding='SAME', use_bias=bias)
        
        ## 5 Input feature : Pool1, 2, 3, 4, 5 
        self.input_spec = [tf.keras.layers.InputSpec(ndim=4), tf.keras.layers.InputSpec(ndim=4), tf.keras.layers.InputSpec(ndim=4), tf.keras.layers.InputSpec(ndim=4), tf.keras.layers.InputSpec(ndim=4)]        
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.groups_list = groups_list
        if self.groups_list == [1]:
            self.groups_list = [1 for _ in range(len(in_channels_list))]
        self.width_mult = max(width_mult_list)
        self.kernel_size = (kernel_size,kernel_size)
        self.filters = max(out_channels_list)
        self.start_point = start_C_1
        
    def build(self, input_shape):
        self.input_dim = input_shape[0][-1]
        
        kernel_shape = self.kernel_size + (max(self.in_channels_list), self.filters)
        
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
    def call(self, input):
        y=[]
        for idx in range(len(width_mult_list)):
            self.in_channels = self.in_channels_list[idx]
            self.out_channels = self.out_channels_list[idx]
            self.groups = self.groups_list[idx]
            self.start = int(self.start_point[idx] * max(self.in_channels_list))
            self.start2 = int(self.start_point[idx] * max(self.out_channels_list))

            weight = self.kernel[:, :, self.start:self.start+self.in_channels, self.start2:self.start2+self.out_channels]

            y.append(tf.nn.conv2d(input[idx], weight, self.strides, padding='SAME')+self.bias[self.start2:self.start2+self.out_channels])

        return y

class SwitchableBatchNorm2d(tf.keras.models.Model):
    def __init__(self,num_features_list):
        super(SwitchableBatchNorm2d, self).__init__()
        self.num_features_list = num_features_list
        self.num_features = max(num_features_list)
        bns = []
        for i in num_features_list:
            bns.append(tf.keras.layers.BatchNormalization(trainable=True))
        self.bn = bns
        self.width_mult = max(width_mult_list)
        self.ignore_model_profiling = True
        
    def call(self, input):
        y=[]
        for idx in range(len(width_mult_list)):
            y.append(self.bn[idx](input[idx]))
        return y

class WeightSharingTransConv2d(tf.keras.layers.Conv2D):
    def __init__(self, in_channels_list, out_channels_list,
                 kernel_size, stride=1, padding=0, dilation=1,
                 groups_list=[1], bias=True):
        
        super(WeightSharingTransConv2d, self).__init__(
            max(out_channels_list), (kernel_size, kernel_size),
            strides=(stride, stride), padding='SAME', use_bias=bias)
        
        ## 5 Compressed Feature : Pool1, 2, 3, 4, 5 
        self.input_spec = [tf.keras.layers.InputSpec(ndim=4), tf.keras.layers.InputSpec(ndim=4), tf.keras.layers.InputSpec(ndim=4), tf.keras.layers.InputSpec(ndim=4), tf.keras.layers.InputSpec(ndim=4)] 
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.groups_list = groups_list
        if self.groups_list == [1]:
            self.groups_list = [1 for _ in range(len(in_channels_list))]
        self.width_mult = max(width_mult_list)
        self.kernel_size = (kernel_size,kernel_size)
        self.filters = max(out_channels_list)
        self.start_point = start_C_1
        
    def build(self, input_shape):
        self.input_dim = input_shape[0][-1]
        self.height = input_shape[0][1]
        self.width = input_shape[0][2]
        kernel_shape = self.kernel_size + (self.filters, max(self.in_channels_list))
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
    def call(self, input_l):
        y=[]
        for idx in range(len(width_mult_list)):
            self.in_channels = self.in_channels_list[idx]
            self.out_channels = self.out_channels_list[idx]
            self.groups = self.groups_list[idx]
            self.start = int(self.start_point[idx] * max(self.out_channels_list))
            self.start2 = int(self.start_point[idx] * max(self.in_channels_list))

            self.batch_size = tf.shape(input_l[idx])[0]
            
            output_shape = [self.batch_size, self.height*2, self.width*2, self.out_channels_list[idx]]
            weight = self.kernel[:, :, self.start:self.start+self.out_channels, self.start2:self.start2+self.in_channels]            
            y.append(tf.nn.conv2d_transpose(input_l[idx], weight, output_shape, self.strides, padding='SAME')+self.bias[self.start:self.start+self.out_channels])


        return y



