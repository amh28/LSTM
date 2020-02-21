import tensorflow as tf
from keras import backend as K
from keras.layers import Layer
from keras.initializers import RandomUniform, glorot_uniform

class MyLstmLayer(tf.keras.layers.Layer):

    def __init__(self, num_outputs):        
        super(MyLstmLayer, self).__init__()
        self.num_outputs = num_outputs
    
    #input_shape -> [batch_size,x,y]
    def build(self, input_shape):        
        self.kernel = self.add_weight("kernel",shape=[int(input_shape[-1]),self.num_outputs], initializer=glorot_uniform(2))        
        super(MyLstmLayer,self).build(input_shape)

    def call(self, input):
        print("kernel",self.kernel)
        print("input", input)
        return tf.matmul(input, self.kernel)
     