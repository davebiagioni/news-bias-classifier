from __future__ import unicode_literals, print_function

import os
import re
import json
import sys
import pickle
import glob
from collections import Counter

import numpy as np
import multiprocessing as mp
import pandas as pd
import spacy

NLP = spacy.load('en')


def get_file_list(datadir, exclude_regex=None):
  '''Get a list of the data files.'''

  files = []
  for dirpath, dirnames, filenames in os.walk(datadir):
    if len(filenames) > 0:
      files.extend([os.path.join(dirpath, x) for x in filenames])
  
  if exclude_regex:
    files = [x for x in files if not re.match(exclude_regex, x)]

  return files

      
def clean_string(string):
  '''Simple preprocessing to remove non-printable characters and excess whitespace.'''

  return re.sub('\s+', ' ', string)


def create_dataframe_serial(files=None):
  '''Create a dataframe from a file list, filtering out non-political articles.'''

  df = pd.DataFrame(columns=['text', 'label', 'url'], data=np.chararray((len(files), 3)))
  
  row = 0
  for filename in files: 

    # Open file.
    with open(filename, 'r') as f:
      data = json.load(f)
    
    # Skip if no taxonomy labels.
    if len(data['taxonomy']) == 0:
      continue

    # Get taxonomy labels and filter on "politics", skipping if none exist.
    labels = [data['taxonomy'][i]['label'] for i in range(len(data['taxonomy']))]
    labels = [x for x in labels if re.match('.*politics', x)]
    if len(labels) == 0:
      continue

    # Populate row, doing basic cleaning of whitespace and non-printable characters
    # in the article text.
    df.loc[row] = [clean_string(data['text']), data['label'], data['url']]

    # Keeping track of the last row we populated.
    row += 1

  # Drop empty rows at tale of dataframe.
  df = df.drop(df.index[row:])
  
  return df


def create_dataframe(files):
  '''Use multiprocessing to create dataframe.'''

  # Initialize output and mp pool
  df = None
  pool = mp.Pool()

  # Split the files manually... this is not clean.
  files = [ map(str, x) for x in np.array_split(files, mp.cpu_count())]

  # Try to create dataframe list using multiprocessing and shut down gracefully
  try:
    df = pool.map(create_dataframe_serial, files)
  except:
    print('multiprocessing error: {}'.format(sys.exc_info()[0]))
  finally:
    pool.close()
    pool.terminate()
    pool.join()

  # Create a single dataframe if processing was successful
  if df:
    df = pd.concat(df, axis=0, ignore_index=True)
  else:
    df = pd.DataFrame()

  return df

# Helper function to parse a document.


def parse_doc_serial(doc, keep_stops=False, min_sents=3):
  '''Return text containing only lemmatized, alphanumeric tokens with POS tag.

  Args:
    doc: spacy parsed doc
    keep_stops:  boolean saying whether we will keep stop words
    min_sents:  minimum number of sentences to parse, else return ''

  Returns parsed string with appended PoS tags.
  '''
  
  def token_formatter(token):
    return '{}_{}'.format(x.lemma_, x.pos_)
  
  # Parse the doc.
  doc = NLP(doc)
  
  # Check that document has at least min_sents sentences.
  num_sents = len([sent for sent in doc.sents])
  if num_sents < min_sents:
    return ''
  
  # Keep alphanumeric, lemmatized tokens with PoS tags.
  if keep_stops:
    text = [token_formatter(x) for x in doc if x.is_alpha]
  else:
    text = [token_formatter(x) for x in doc if x.is_alpha and not x.is_stop]
    
  return ' '.join(text)


def parse_docs_mp(args):
  ''' Multiprocessing handler.'''
  return parse_doc_serial(args[0], args[1], args[2])


def parse_docs(text_list, keep_stops, min_sents):
  '''Helper function for multiprocessing.'''

  result = []
  pool = mp.Pool()

  args = zip(text_list, [keep_stops]*len(text_list), [min_sents]*len(text_list))
  print('processing {} docs'.format(len(args)))

  try:
    result = pool.map(parse_docs_mp, args)
  except:
    print('multiprocessing error: {}'.format(sys.exc_info()[0]))
  finally:
    pool.close()
    pool.terminate()
    pool.join()

  return result


def create_vocab(text_list):

  c = Counter()
  for item in text_list:
    c.update(item.split())
    
  print('dictionary size: {}'.format(len(c)))

  vocab_list = [(k, v) for k,v in c.iteritems()]
  vocab_list.sort(key=lambda x: -1 * x[1])

  vocab_word2idx = {x[0]: ix for ix,x in enumerate(vocab_list)}
  vocab_idx2word = {v: k for k,v in vocab_word2idx.iteritems()}

  return vocab_list, vocab_word2idx, vocab_idx2word


def write_dataset(filepath, df, keep_stops, min_sents, vocab, word2idx, idx2word):
  '''Helper to write dataset and configuration params to pickle.'''

  basename = '{}-{}-{}.pkl'.format(os.path.basename(filepath), keep_stops, min_sents)
  filepath = os.path.join(os.path.dirname(filepath), basename)

  data = {
    'df': df,
    'keep_stops': keep_stops,
    'min_sents': min_sents,
    'vocab': vocab,
    'word2idx': word2idx,
    'idx2word': idx2word
  }

  with open(filepath, 'w') as f:
    _ = pickle.dump(data, f)
    print('wrote to {}'.format(filepath))


def read_dataset(filepath):

  with open(filepath, 'r') as f:
    data = pickle.load(f)

  df = data['df']
  keep_stops = data['keep_stops']
  min_sents = data['min_sents']
  vocab = data['vocab']
  word2idx = data['word2idx']
  idx2word = data['idx2word']

  return df, keep_stops, min_sents, vocab, word2idx, idx2word


def filter_top_words(lst, top_words):
  '''Given a list of lists of vocab indices, filter out those not in top n'''

  if top_words:
    for ix in xrange(len(lst)):
      lst[ix] = filter(lambda x: x < top_words, lst[ix])

  return lst

def create_model_checkpoint(basename):

  filepath = basename + '_{epoch:03d}_{val_loss:.5f}_{val_acc:.5f}.h5'

  return os.path.join(basename, filepath), os.path.join(basename, basename + '.h5')
