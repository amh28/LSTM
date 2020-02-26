import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer


class MyLSTMCell(Layer):
    def __init__(self, num_cells):
        super(MyLSTMCell,self).__init__()        
        self.num_cells = num_cells
        
    #input_size = [sequence lenght, units]
    def build(self, input_size):
        self.k_f = self.add_weight(name='k_f',shape=[input_size[-1],
                                         self.num_cells], initializer='uniform')
        self.k_i = self.add_weight(name='k_i',shape=[input_size[-1],self.num_cells],
                                                                initializer='uniform')
        self.k_c = self.add_weight(name='k_c',shape=[input_size[-1], self.num_cells],
                                                                initializer='uniform')
        self.k_o = self.add_weight(name='k_o',shape=[input_size[-1], self.num_cells],
                                                                initializer='uniform')
        self.recurrent_kf = self.add_weight(name='recurrent_kf',
                                                    shape=[self.num_cells, self.num_cells], initializer='uniform')
        self.recurrent_ki = self.add_weight(name='recurrent_ki',
                                                    shape=[self.num_cells, self.num_cells], initializer='uniform')                                                            
        self.recurrent_kc = self.add_weight(name='recurrent_kc',
                                                    shape=[self.num_cells, self.num_cells], initializer='uniform')                                                                                                                
        self.recurrent_ko = self.add_weight(name='recurrent_ko',
                                                    shape=[self.num_cells, self.num_cells], initializer='uniform')           
        super(MyLSTMCell, self).build(input_size)

    #states = [batch_size,1,self.num_cells]
    def call(self, inputs, cells, states, log=False):        
        prev_stat = states[:,-1,:]
        prev_cell = cells[:,-1,:]
        #print("recurrent shape: {}",prev_stat.shape)
        input = K.dot(inputs,self.k_i) + K.dot(prev_stat,self.recurrent_ki)
        forget = K.dot(inputs,self.k_f) + K.dot(prev_stat,self.recurrent_kf)       
        output = tf.sigmoid(K.dot(inputs,self.k_o))
        update = tf.tanh(K.dot(inputs,self.k_c))
        
        input = tf.sigmoid(input)
        forget = tf.sigmoid(forget)
        output = tf.sigmoid(output)
        update = tf.tanh(update)        
        
        c1 = tf.multiply(forget,prev_cell)
        c2 = tf.multiply(input,update)
        cell = c1 + c2
        hidden = tf.multiply(output,tf.tanh(cell))        

        return cell,hidden
        #return [],[]