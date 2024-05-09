import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
from keras.models import load_model

def tokenize(train,val):
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(train)
  tr_seq = tokenizer.texts_to_sequences(train)
  val_seq = tokenizer.texts_to_sequences(val)
  max_len = max([len(seq) for seq in tr_seq])
  vocab_size = len(tokenizer.word_index) + 1
  
  return tr_seq, val_seq, max_len, vocab_size

def summary(txt):
    input = input_tok.texts_to_sequences([txt])
    input = pad_sequences(input, maxlen=max_input_len, padding='post')
    target = np.zeros((1, max_target_len), dtype='int')
    target[0,0] = target_tok.word_index.get('<sos>', 0) 
    summary = ''
     
    stop = False
    while not stop:
        outputs = model.predict([input, target])
        pred_idx = np.argmax(outputs[0, 0, :])
        pred_word = None
        for word, idx in target_tok.word_index.items():
            if idx == pred_idx:
                summary += ' {}'.format(word)
                pred_word = word
        
        if pred_word == '<eos>' or len(summary.split()) > max_target_len:
            stop = True
        target[0, 0] = pred_idx
    
    return summary

data = pd.read_csv("cnn_1000.csv")
data["text"].fillna("", inplace=True)
data["summary"].fillna("", inplace=True)

x_tr, x_val, y_tr, y_val = train_test_split(
    np.array(data["text"]),np.array(data["summary"]),
    test_size=0.1,random_state=0,shuffle=True)
  
# Tokenize input texts and target texts
x_tr_seq, x_val_seq, max_input_len,input_vocab_size = tokenize(x_tr, x_val) 
y_tr_seq, y_val_seq, max_target_len,target_vocab_size= tokenize(y_tr, y_val) 

# Pad sequences to make input and target uniform in length
enc_input_tr = pad_sequences(x_tr_seq, maxlen=max_input_len, padding='post')
enc_input_val = pad_sequences(x_val_seq, maxlen=max_input_len, padding='post')
dec_input_tr = pad_sequences(y_tr_seq, maxlen=max_target_len, padding='post')
dec_input_val = pad_sequences(y_val_seq, maxlen=max_target_len, padding='post')

# Shift target sequences by one position
dec_target_tr = np.zeros_like(dec_input_tr)
dec_target_tr[:, 0:-1] = dec_input_tr[:, 1:]
dec_target_tr[:, -1] = 0
dec_target_val = np.zeros_like(dec_input_val)
dec_target_val[:, 0:-1] = dec_input_val[:, 1:]
dec_target_val[:, -1] = 0

# Define the encoder-decoder model
latent_dim = 200
enc_inputs = Input(shape=(None,))
enc_emb = Embedding(input_vocab_size, latent_dim, mask_zero=True)(enc_inputs)
enc_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
enc_outputs, state_h, state_c = enc_lstm(enc_emb)

dec_inputs = Input(shape=(None,))
dec_emb = Embedding(target_vocab_size, latent_dim, mask_zero=True)(dec_inputs)
dec_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
dec_outputs, _, _ = dec_lstm(dec_emb, initial_state=[state_h, state_c])
dec_dense = Dense(target_vocab_size, activation='softmax')
dec_outputs = dec_dense(dec_outputs)

model = Model([enc_inputs, dec_inputs], dec_outputs)
model.summary()

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit([enc_input_tr, dec_input_tr], dec_target_tr,
          validation_data=([enc_input_val, dec_input_val], dec_target_val),
          batch_size=64, epochs=50)

# Save the model optionally
#model.save("encoder_decoder_model.h5")
#model = load_model("encoder_decoder_model.h5")

input_tok = Tokenizer()
input_tok.fit_on_texts(x_val)
target_tok = Tokenizer()
target_tok.fit_on_texts(y_val)
for i in range(len(x_val)):
  print("Article:", x_val[i])
  print("Reference summary:", y_val[i])
  print("Candidate summary:",summary(x_val[i]))