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
        self.k_bf = self.add_weight(name='k_bf',shape=[self.num_cells], initializer='zeros')
        self.k_bi = self.add_weight(name='k_bi',shape=[self.num_cells], initializer='zeros')
        self.k_bc = self.add_weight(name='k_bc',shape=[self.num_cells], initializer='zeros')
        self.k_bo = self.add_weight(name='k_bo',shape=[self.num_cells], initializer='zeros')

        self.recurrent_kf = self.add_weight(name='recurrent_kf',
                                                    shape=[self.num_cells, self.num_cells], initializer='uniform')
        self.recurrent_ki = self.add_weight(name='recurrent_ki',
                                                    shape=[self.num_cells, self.num_cells], initializer='uniform')                                                            
        self.recurrent_kc = self.add_weight(name='recurrent_kc',
                                                    shape=[self.num_cells, self.num_cells], initializer='uniform')                                                                                                                
        self.recurrent_ko = self.add_weight(name='recurrent_ko',

        self.rk_bf = self.add_weight(name='rk_bf',shape=[self.num_cells], initializer='zeros')
        self.rk_bi = self.add_weight(name='rk_bi',shape=[self.num_cells], initializer='zeros')
        self.rk_bc = self.add_weight(name='rk_bc',shape=[self.num_cells], initializer='zeros')
        self.rk_bo = self.add_weight(name='rk_bo',shape=[self.num_cells], initializer='zeros')

        super(MyLSTMCell, self).build(input_size)

    #states = [batch_size,1,self.num_cells]
    def call(self, inputs, cells, states, log=False):        
        prev_stat = states[:,-1,:]
        prev_cell = cells[:,-1,:]
        #print("recurrent shape: {}",prev_stat.shape)
        input = tf.add(K.dot(inputs,self.k_i),self.k_bi) + tf.add(K.dot(prev_stat,self.recurrent_ki),self.rk_bi)
        forget = tf.add(K.dot(inputs,self.k_f),self.k_bf) + tf.add(K.dot(prev_stat,self.recurrent_kf),self.rk_bf)       
        output = tf.sigmoid(tf.add(K.dot(inputs,self.k_o),self.k_bo))
        update = tf.tanh(tf.add(K.dot(inputs,self.k_c),self.k_bc))
        
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