import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer


class MyLSTMCell(Layer):
    def __init__(self, num_cells):
        super(MyRNNCell,self).__init__()        
        self.num_cells = num_cells
        
    #input_size = [sequence lenght, units]
    def build(self, input_size):
        self.kernel = self.add_weight(name='kernel',shape=[input_size[-1],
                                         self.num_cells], initializer='uniform')
        self.recurrent_kernel = self.add_weight(name='recurrent_kernel',
                                                    shape=[self.num_cells, self.num_cells], initializer='uniform')        
        super(MyLSTMCell, self).build(input_size)

    #states = [batch_size,1,self.num_cells]
    def call(self, inputs, cell, states, log=False):
        prev_stat = states[:,-1,:]
        xh = K.dot(inputs,self.kernel) 
        hh = K.dot(prev_stat,self.recurrent_kernel)
        h = xh + hh      
        fh =  tf.sigmoid(h) 
        return fh