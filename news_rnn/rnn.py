from keras.models import Sequential, load_model
from keras.layers import Dense, GRU, Bidirectional, Activation, Dropout
from keras.layers.core import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import nadam
from keras.layers.normalization import BatchNormalization

import numpy as np

def data_gen(df, batch_size, frac_drop, maxlen, topn, validation=False):

  def drop(i, n):
    drop_idx = np.random.choice(i, n, replace=False)
    _ = [i.remove(y) for y in drop_idx]
    return i

  indexes = range(df.shape[0])
  curr = 0
  while True:
    X, Y = [], []
    for batch_idx in range(batch_size):
      if curr == len(indexes):
        np.random.shuffle(indexes)
        curr = 0
      x = df['encoded_text'].iloc[indexes[curr]]
      if not validation and frac_drop > 0.001:
        idx = drop(range(len(x)), int(frac_drop * len(x)))
        x = [x[i] for i in idx]
      X.append(x)
      Y.append(np.array(df['encoded_label'].iloc[indexes[curr]]))
      curr += 1
    X = pad_sequences(X, maxlen=maxlen, value=topn,  
                      padding='post', truncating='post')  
    Y = np.array(Y).reshape((batch_size, 1))
    yield (X, Y)


def data_getter(df, maxlen, topn):

  x = pad_sequences(df['encoded_text'].tolist(), maxlen=maxlen, value=topn,
                   padding='post', truncating='post')
  y = np.array(df['encoded_label'])

  return x, y


def get_training_model(topn, embed_dim, dense_dim, gru_dim, num_gru, maxlen, dropout, 
  bidirectional, lr=0.001):

  model = Sequential()

  model.add(Embedding(topn+1, embed_dim, input_length=maxlen))

  for i in range(num_gru):
    
    if i < (num_gru-1):
      return_sequences = True
    else:
      return_sequences = False

    gru_layer = GRU(gru_dim, dropout=dropout, recurrent_dropout=dropout, 
      return_sequences=return_sequences)

    if bidirectional:
      gru_layer = Bidirectional(gru_layer)

    model.add(gru_layer)
  
  model.add(Dropout(dropout))
  model.add(Dense(dense_dim, activation='relu'))

  model.add(Dropout(dropout))
  model.add(Dense(1, activation='sigmoid'))

  opt = nadam(lr=lr)
  _ = model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
  
  return model


def split_model_layers(model, topn, embed_dim, dense_dim, gru_dim, num_gru, maxlen, bidirectional):

  in_model = Sequential()

  in_model.add(Embedding(topn+1, embed_dim, input_length=maxlen, 
    weights=model.layers[0].get_weights()))

  for i in range(num_gru):
    
    gru_layer = GRU(gru_dim, weights=model.layers[i+1].get_weights(), 
      return_sequences=True)

    if bidirectional:
      gru_layer = Bidirectional(gru_layer)

    in_model.add(gru_layer)

  out_model = Sequential()

  dense_input_dim = gru_dim
  if bidirectional:
    dense_input_dim *= 2

  out_model.add(Dense(dense_dim, weights=model.layers[num_gru+2].get_weights(), activation='relu',
    input_dim=dense_input_dim))

  out_model.add(Dense(1, weights=model.layers[num_gru+4].get_weights(), activation='sigmoid',
    input_dim=dense_dim))

  return in_model, out_model

def evaluate_sequential_probs(X, idx, in_model, out_model):
  embeddings = in_model.predict_proba(X[idx:idx+1, :], verbose=0)
  return out_model.predict_proba(embeddings.squeeze(), verbose=0).squeeze()


