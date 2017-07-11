from keras.models import Sequential, load_model
from keras.layers import Dense, GRU, Bidirectional, Activation, Dropout
from keras.layers.core import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import nadam
from keras.layers.normalization import BatchNormalization

import numpy as np


DEFAULT_GRU_ACTIVATION = 'tanh'


def get_training_model(topn, embed_dim, dense_dim, gru_dim, num_gru, maxlen, dropout, 
  bidirectional, output_size, gru_activation=None, lr=0.001):

  if not gru_activation:
    gru_activation = DEFAULT_GRU_ACTIVATION

  model = Sequential()

  model.add(Embedding(topn+1, embed_dim, input_length=maxlen))

  for i in range(num_gru):
    
    if i < (num_gru-1):
      return_sequences = True
    else:
      return_sequences = False

    gru_layer = GRU(gru_dim, dropout=dropout, recurrent_dropout=dropout, 
      return_sequences=return_sequences, activation=gru_activation)

    if bidirectional:
      gru_layer = Bidirectional(gru_layer)

    model.add(gru_layer)
  
  model.add(Dropout(dropout))
  model.add(Dense(dense_dim, activation='relu'))

  if output_size == 1:
    activation = 'sigmoid'
    loss = 'binary_crossentropy'
  else:
    activation = 'softmax'
    loss = 'categorical_crossentropy'

  model.add(Dropout(dropout))
  model.add(Dense(output_size, activation=activation))

  opt = nadam(lr=lr)
  _ = model.compile(loss=loss, optimizer=opt, metrics=['acc'])
  
  return model


def split_model_layers(model, topn, embed_dim, dense_dim, gru_dim, num_gru, maxlen, 
  output_size, bidirectional, gru_activation=None):

  if not gru_activation:
    gru_activation = DEFAULT_GRU_ACTIVATION

  # INPUT LAYERS
  in_model = Sequential()

  in_model.add(Embedding(topn+1, embed_dim, input_length=maxlen, 
    weights=model.layers[0].get_weights()))

  for i in range(num_gru):
    
    gru_layer = GRU(gru_dim, weights=model.layers[i+1].get_weights(), 
      return_sequences=True, activation=gru_activation)

    if bidirectional:
      gru_layer = Bidirectional(gru_layer)

    in_model.add(gru_layer)

  # OUTPUT LAYERS
  out_model = Sequential()

  dense_input_dim = gru_dim
  if bidirectional:
    dense_input_dim *= 2

  out_model.add(Dense(dense_dim, weights=model.layers[num_gru+2].get_weights(), activation='relu',
    input_dim=dense_input_dim))

  if output_size == 1:
    activation = 'sigmoid'
  else:
    activation = 'softmax'

  out_model.add(Dense(output_size, weights=model.layers[num_gru+4].get_weights(), activation=activation,
    input_dim=dense_dim))
  
  return in_model, out_model


def sequential_pred_for_class(X, df, idx, in_model, out_model):
  embeddings = in_model.predict(X[idx:(idx+1), :], verbose=0).squeeze()
  return out_model.predict(embeddings)[:, df['encoded_domain'].iloc[idx]]


