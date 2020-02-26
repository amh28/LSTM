from Layer import MyLSTMCell
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer

        
      
class LSTM(Layer):
    def __init__(self, num_cells):
        super(LSTM,self).__init__()        
        self.num_cells = num_cells        
    
    #[batch_size, sequence length, units]
    def build(self, input_size):
        self.lstmCell = MyLSTMCell.MyLSTMCell(self.num_cells)
        super(LSTM, self).build(input_size)
       
    
    #I do one call per batch
    def call(self, inputs):
        batch_size = inputs.shape[0]
        timesteps = inputs.shape[1]
        states = tf.zeros([batch_size,1,self.num_cells])
        cells = tf.zeros([batch_size,1,self.num_cells])
        
        for t in range(0,timesteps):
            #print("timestep ",t)                    
            cell, hidden = self.lstmCell(inputs= inputs[:,t,:],cells=cells,states=states)            
            cell = K.expand_dims(cell,1)
            hidden = K.expand_dims(hidden,1)
            states = tf.concat([states,hidden],1)
            cells = tf.concat([cells,cell],1)
            
        states = states[:,1:,:]        
        cells = cells[:,1:,:]        
        return states
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1],self.num_cells)
        


        
        


