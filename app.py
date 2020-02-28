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

val_every = 50


def get_batch(vectorized_songs, seq_length, batch_size):
    n = vectorized_songs.shape[0] - 1
    idx = np.random.choice(n-seq_length, batch_size)  
    input_batch = [vectorized_songs[i : i+seq_length] for i in idx]
    output_batch = [vectorized_songs[i+1 : i+seq_length+1] for i in idx]
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
    input_eval = [char2idx[s] for s in start_string] # TODO
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    model.reset_states()
    #tqdm._instances.clear()

    for i in tqdm(range(generation_length)):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id]) # TODO 
    return (start_string + ''.join(text_generated))


@tf.function
def val_step(x, y,model): 
    y_hat = model(x)
    loss = compute_loss(y, y_hat)    
    return loss

@tf.function
def train_step(x, y,model,optimizer): 
    # Use tf.GradientTape()
    with tf.GradientTape() as tape:
        y_hat = model(x) # TODO
        loss = compute_loss(y, y_hat) # TODO
    
    grads = tape.gradient(loss, model.trainable_weights) # TODO     
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss

def train(model_class, train_data, val_data,num_training_iterations,learning_rate, vocab_size,embedding_dim,rnn_units,batch_size,seq_length,checkpoint_prefix):
    model = build_model(model_class,vocab_size, embedding_dim, rnn_units, batch_size,seq_length)
    model.summary()
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    history = []
    val = []
    plotter = mdl.util.PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Loss')
    
    if hasattr(tqdm, '_instances'): tqdm._instances.clear() # clear if it exists

    for iter in tqdm(range(num_training_iterations)):
        x_batch, y_batch = get_batch(train_data, seq_length, batch_size)
        loss = train_step(x_batch, y_batch,model,optimizer)

        # Update the progress bar
        history.append((iter,loss.numpy().mean()))
        plotter.plot(history)

        # Perform validation
        if iter % val_every == 0:            
            x_val, y_val = get_batch(val_data,seq_length,batch_size)
            loss_val = val_step(x_val, y_val,model)          
            val.append((iter,loss_val.numpy().mean()))

            h, = plt.plot(*zip(*history), label='Train')
            v, = plt.plot(*zip(*val), label='Val')
            plt.legend(handles=[h, v])
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            ipythondisplay.clear_output(wait=True)
            ipythondisplay.display(plt.gcf())

        # Update the model with the changed weights!
        #if iter % 50 == 0:                 
        #    print("iter {}, loss: {}".format(iter,loss.numpy().mean()))            
        #    model.save_weights(checkpoint_prefix)

    # Save the trained model and the weights
    #model.save_weights(checkpoint_prefix)

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

def get_songs():
    songs = mdl.lab1.load_training_data()    
    #mdl.lab1.play_song(example_song)
    #train 60 val 20
    train_songs = songs[:60]
    val_songs = songs[60:]

    all_songs = "\n\n".join(songs)
    train_joined = "\n\n".join(train_songs) 
    val_joined = "\n\n".join(val_songs)
    
    vocab = sorted(set(all_songs))   
    char2idx = {u:i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)
    vectorized_trained = np.array([char2idx[char] for char in train_joined])
    vectorized_val = np.array([char2idx[char] for char in val_joined])

    return char2idx, idx2char,vectorized_trained, vectorized_val 

def main():
    
    batch_size=4 # Experiment between 1 and 64
    seq_length=100 # Experiment between 50 and 500
    rnn_units=1024 # Experiment between 1 and 2048
    learning_rate = 5e-3  # Experiment between 1e-5 and 1e-1
    sample_num=10000     
    num_training_iterations = 2# Increase this to train longer
    
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")
    
    vocab_size=83
    embedding_dim=256    
    char2idx, idx2char, train_data, val_data = get_songs()
    ##########
    #Training    
    train(LSTM,train_data,val_data,num_training_iterations,learning_rate, vocab_size,embedding_dim,rnn_units,batch_size,seq_length,checkpoint_prefix)
    
    ##########
    #Testing
    ##########    
    #test(LSTM,char2idx,idx2char,vocab_size,embedding_dim,rnn_units,batch_size,seq_length,sample_num,checkpoint_prefix)    

if __name__ == "__main__":
    main()