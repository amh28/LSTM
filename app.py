from Layer import MyLstmLayer
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

def main():
    layer = MyLstmLayer.MyLstmLayer(4)    
    output = layer(tf.zeros([2,5]))
    print("output ",output)



if __name__ == "__main__":
    main()