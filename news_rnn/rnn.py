from keras.models import Sequential, load_model
from keras.layers import Dense, GRU, Bidirectional, Activation, Dropout
from keras.layers.core import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences

import numpy as np

def data_gen(df, frac_drop, maxlen, topn, validation=False):

  def drop(i, n):
    drop_idx = np.random.choice(i, n, replace=False)
    _ = [i.remove(y) for y in drop_idx]
    return i

  indexes = range(df.shape[0])
  curr = 0
  while True:
    if curr == len(indexes):
      np.random.shuffle(indexes)
      curr = 0
    x = df['encoded_text'].iloc[indexes[curr]]
    if not validation:
      idx = drop(range(len(x)), int(frac_drop * len(x)))
      x = [x[i] for i in idx]
    x = pad_sequences([x], maxlen=maxlen, value=topn, 
                      padding='post', truncating='post')
    y = np.array(df['encoded_label'].iloc[indexes[curr]]).reshape((1, 1))
    curr += 1
    yield (x, y)


def get_training_model(topn, embed_dim, dense_dim, gru_dim, maxlen, dropout, 
  bidirectional):

  model = Sequential()

  #model.add(Dropout(input_dropout, input_shape=maxlen))

  model.add(Embedding(topn+1, embed_dim, input_length=maxlen))
  #model.add(Embedding(topn+1, embed_dim))

  gru_layer = GRU(gru_dim, dropout=dropout, recurrent_dropout=dropout)

  if bidirectional:
    gru_layer = Bidirectional(gru_layer)

  model.add(gru_layer)
    
  model.add(Dropout(dropout))
  model.add(Dense(dense_dim, activation='relu'))

  model.add(Dropout(dropout))
  model.add(Dense(1, activation='sigmoid'))

  _ = model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
  
  return model


def split_model_layers(model, topn, embed_dim, dense_dim, gru_dim, maxlen, bidirectional):

  in_model = Sequential()

  in_model.add(Embedding(topn+1, embed_dim, input_length=maxlen, 
    weights=model.layers[0].get_weights()))

  gru_layer = GRU(gru_dim, weights=model.layers[1].get_weights(), return_sequences=True)

  if bidirectional:
    gru_layer = Bidirectional(gru_layer)

  in_model.add(gru_layer)

  out_model = Sequential()

  dense_input_dim = gru_dim
  if bidirectional:
    dense_input_dim *= 2

  out_model.add(Dense(dense_dim, weights=model.layers[3].get_weights(), activation='relu',
    input_dim=dense_input_dim))

  out_model.add(Dense(1, weights=model.layers[5].get_weights(), activation='sigmoid',
    input_dim=dense_dim))

  return in_model, out_model

def evaluate_sequential_probs(X, idx, in_model, out_model):
  embeddings = in_model.predict_proba(X[idx:idx+1, :], verbose=0)
  return out_model.predict_proba(embeddings.squeeze(), verbose=0).squeeze()


