from Layer import MyRNNCell
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer

        
      
class RNN(Layer):
    def __init__(self, num_cells):
        super(RNN,self).__init__()        
        self.num_cells = num_cells        
    
    #[batch_size, sequence length, units]
    def build(self, input_size):
        self.rnnCell = MyRNNCell.MyRNNCell(self.num_cells)
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
        


        
        


