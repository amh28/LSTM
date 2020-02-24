import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Layer

class MyRNNCell(Layer):
    def __init__(self, num_cells):
        super(MyRNNCell,self).__init__()        
        self.num_cells = num_cells
        
    #input_size = [sequence lenght, units]
    def build(self, input_size):
        self.kernel = self.add_weight(name='kernel',shape=[input_size[-1],
                                         self.num_cells], initializer='uniform')
        self.recurrent_kernel = self.add_weight(name='recurrent_kernel',
                                                    shape=[self.num_cells, self.num_cells], initializer='uniform')        
        #super(MyRNNCell, self).build(input_size)
        self.built=True

    #states = [batch_size,1,self.num_cells]
    def call(self, inputs, states, log=False):
        prev_stat = states[:,-1,:]
        xh = K.dot(inputs,self.kernel) 
        hh = K.dot(prev_stat,self.recurrent_kernel)
        h = xh + hh      
        fh =  [ tf.sigmoid(h) ]
        return fh
        
      
class RNN(Layer):
    def __init__(self, num_cells):
        super(RNN,self).__init__()        
        self.num_cells = num_cells        
    
    #[batch_size, sequence length, units]
    def build(self, input_size):
        self.rnnCell = MyRNNCell(self.num_cells)
        #super(RNN, self).build(input_size)
        self.built=True
    
    #I do one call per batch
    def call(self, inputs):
        batch_size = inputs.shape[0]
        timesteps = inputs.shape[1]
        states = tf.zeros([batch_size,1,self.num_cells])
        
        for t in range(0,timesteps):
            #print("timestep ",t)                    
            s = self.rnnCell(inputs= inputs[:,t,:],states=states) 
            s = K.expand_dims(s,1)
            states = tf.concat([states,s],1)
            #print("s shape: ",s.shape)            
        states = states[:,1:,:]        
        return states
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1],self.num_cells)
        


        
        


