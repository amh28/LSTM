import os
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.python.keras.layers import Dense, Input, Embedding
from tensorflow.python.keras.models import Model, Sequential

from Layer import RNN
from Layer import LSTM

import mitdeeplearning as mdl
import numpy as np
import time
import functools
from IPython import display as ipythondisplay
from tqdm import tqdm



def get_batch(vectorized_songs, seq_length, batch_size):
    # the length of the vectorized songs string
    n = vectorized_songs.shape[0] - 1
    # randomly choose the starting indices for the examples in the training batch
    idx = np.random.choice(n-seq_length, batch_size)  
    input_batch = [vectorized_songs[i : i+seq_length] for i in idx]
    # input_batch = # TODO
    output_batch = [vectorized_songs[i+1 : i+seq_length+1] for i in idx]
    # output_batch = # TODO
    x_batch = np.reshape(input_batch, [batch_size, seq_length])
    y_batch = np.reshape(output_batch, [batch_size, seq_length])
    return x_batch, y_batch

def build_model(model_class,vocab_size, embedding_dim, rnn_units, batch_size, t):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, t]))
    model.add(LSTM.LSTM(rnn_units))
    model.add(Dense(vocab_size))
    return model

def compute_loss(labels, logits):
    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)    
    return loss

def generate_text(start_string,char2idx,idx2char,model, generation_length=1000):
    # Evaluation step (generating ABC text using the learned RNN model)

    input_eval = [char2idx[s] for s in start_string] # TODO
    # input_eval = ['''TODO''']
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Here batch size == 1
    model.reset_states()
    #tqdm._instances.clear()

    for i in tqdm(range(generation_length)):
        predictions = model(input_eval)
        # predictions = model('''TODO''')
      
        # Remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        '''TODO: use a multinomial distribution to sample'''
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        # predicted_id = tf.random.categorical('''TODO''', num_samples=1)[-1,0].numpy()

        # Pass the prediction along with the previous hidden state
        #   as the next inputs to the model
        input_eval = tf.expand_dims([predicted_id], 0)

        '''TODO: add the predicted character to the generated text!'''
        # Hint: consider what format the prediction is in vs. the output
        text_generated.append(idx2char[predicted_id]) # TODO 
        # text_generated.append('''TODO''')
    return (start_string + ''.join(text_generated))


@tf.function
def train_step(x, y,model,optimizer): 
    # Use tf.GradientTape()
    with tf.GradientTape() as tape:
        y_hat = model(x) # TODO
        loss = compute_loss(y, y_hat) # TODO
    
    grads = tape.gradient(loss, model.trainable_weights) # TODO     
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss

def test(model_class,char2idx,idx2char,vocab_size,embedding_dim,rnn_units,batch_size,seq_length,sample_num,checkpoint_prefix):
    model = build_model(model_class,vocab_size, embedding_dim, rnn_units, batch_size,seq_length) # TODO
    model.load_weights(checkpoint_prefix)
    model.build(tf.TensorShape([1, None]))
    model.summary()

    generated_text = generate_text("X",char2idx, idx2char, model,sample_num) # TODO
    generated_songs = mdl.lab1.extract_song_snippet(generated_text)
    
    for i, song in enumerate(generated_songs): 
    # Synthesize the waveform from a song
        waveform = mdl.lab1.play_song(song)

        # If its a valid song (correct syntax), lets play it! 
        if waveform:
            print("Generated song", i)
            ipythondisplay.display(waveform)

def train(model_class, vectorized_songs,num_training_iterations,learning_rate, vocab_size,embedding_dim,rnn_units,batch_size,seq_length,checkpoint_prefix):
    model = build_model(model_class,vocab_size, embedding_dim, rnn_units, batch_size,seq_length)
    model.summary()
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    history = []
    plotter = mdl.util.PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Loss')
    if hasattr(tqdm, '_instances'): tqdm._instances.clear() # clear if it exists

    for iter in tqdm(range(num_training_iterations)):
        # Grab a batch and propagate it through the network
        x_batch, y_batch = get_batch(vectorized_songs, seq_length, batch_size)
        loss = train_step(x_batch, y_batch,model,optimizer)

        # Update the progress bar
        history.append(loss.numpy().mean())
        plotter.plot(history)

        # Update the model with the changed weights!
        if iter % 50 == 0:     
            print("iter {}, loss: {}".format(iter,loss.numpy().mean()))
            model.save_weights(checkpoint_prefix)

    # Save the trained model and the weights
    model.save_weights(checkpoint_prefix)

    
'''model = build_model(len(vocab), embedding_dim=embedding_dim, rnn_units=rnn_units, batch_size=batch_size,t=seq_length)
pred = model(x)
print("Prediction shape: ", pred.shape, "# (batch_size, sequence_length, vocab_size)")
model.summary()'''

'''sampled_indices = tf.random.categorical(pred[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1)
#print(sampled_indices)

print("Input: \n", repr("".join(idx2char[_x[0]])))
#print()
print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices])))
example_batch_loss = compute_loss(y, pred)

print("Prediction shape: ", pred.shape, " # (batch_size, sequence_length, vocab_size)") 
print("scalar_loss:      ", example_batch_loss.numpy().mean())'''


def main():
    songs = mdl.lab1.load_training_data()
    example_song = songs[0]
    #mdl.lab1.play_song(example_song)

    vocab_size=83
    embedding_dim=256
    batch_size=32 # Experiment between 1 and 64
    seq_length=100 # Experiment between 50 and 500
    rnn_units=1024 # Experiment between 1 and 2048

    songs_joined = "\n\n".join(songs) 
    vocab = sorted(set(songs_joined))   
    char2idx = {u:i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)
    vectorized_songs = np.array([char2idx[char] for char in songs_joined])
    
    x, y = get_batch(vectorized_songs, seq_length=seq_length, batch_size=batch_size)
    print("Input shape:      ", x.shape, " # (batch_size, sequence_length)")
    
    num_training_iterations = 2000# Increase this to train longer
    batch_size = 4  
    seq_length = 100
    learning_rate = 5e-3  # Experiment between 1e-5 and 1e-1
    sample_num=10000     

    # Checkpoint location: 
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")
    
    ##########
    #Training
    ##########
    train(LSTM,vectorized_songs,num_training_iterations,learning_rate, vocab_size,embedding_dim,rnn_units,batch_size,seq_length,checkpoint_prefix)
    
    ##########
    #Testing
    ##########    
    #test(LSTM,char2idx,idx2char,vocab_size,embedding_dim,rnn_units,batch_size,seq_length,sample_num,checkpoint_prefix)
    

if __name__ == "__main__":
    main()