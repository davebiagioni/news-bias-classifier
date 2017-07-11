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
  '''Get a list of the data files.

  Args:
    datadir:  directory where data lives.
    exclude_regex: exclude files that match this regular expression.

  Returns:  List of file names.
  '''

  files = []
  for dirpath, dirnames, filenames in os.walk(datadir):
    if len(filenames) > 0:
      files.extend([os.path.join(dirpath, x) for x in filenames])
  
  if exclude_regex:
    files = [x for x in files if not re.match(exclude_regex, x)]

  return files

      
def clean_string(string):
  '''Simple preprocessing to remove non-printable characters and excess whitespace.

  Args:
    string:  a string

  Returns:  String with excess whitespace removed.
  '''

  return re.sub('\s+', ' ', string)


def create_dataframe_serial(args):
  '''Create a dataframe from a file list using the specified JSON field.

  Args:
    args:  tuple containing (filename, input field)

  Returns: pandas data frame
  '''

  # Unpack arguments
  files, field = args

  # Instantiate dataframe
  df = pd.DataFrame(columns=[field, 'label', 'url'], data=np.chararray((len(files), 3)))
  
  row = 0
  for filename in files: 

    # Open file.
    with open(filename, 'r') as f:
      data = json.load(f)
    
    # Populate row, doing basic cleaning of whitespace and non-printable characters
    # in the article text.
    df.loc[row] = [clean_string(data[field]), data['label'], data['url']]

    # Keeping track of the last row we populated.
    row += 1

  # Drop empty rows at tale of dataframe.
  df = df.drop(df.index[row:])
  
  return df


def create_dataframe(files, field='title'):
  '''Use multiprocessing to create a dataframe with cleaned articles.

  Args:
    files:  file list

  Returns: pandas data frame
  '''

  # Initialize output and mp pool
  df = None
  pool = mp.Pool(mp.cpu_count()-1)

  # Split the files manually... this is not clean.
  files = [ map(str, x) for x in np.array_split(files, mp.cpu_count())]
  args = zip(files, [field]*len(files))

  # Try to create dataframe list using multiprocessing and shut down gracefully
  try:
    df = pool.map(create_dataframe_serial, args)
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


def get_shingles(string, size):
    '''Get shingles of a given size from a string.'''

    string = re.sub('\s+', ' ', string)  # remove excess white space

    tokens = string.split()
    num = max(1, len(tokens) - size + 1)
    shingles = [' '.join(tokens[i:i+size]) for i in range(num)]

    return shingles


def find_common_shingles(titles, size, thresh=0.33, min_titles=50):
  '''Returns a list of shingles that are common across all titles.'''
  
  if len(titles) < min_titles:
    return []
  c = Counter()
  title_shingles = [get_shingles(t, size) for t in titles]
  for t in title_shingles:
    local_dict = {}
    for s in t:
      if s not in local_dict:
        c.update([s])
        local_dict[s] = ''
  common = []
  for item in c.most_common(100):
    if (item[1] >= 1. * thresh * len(titles)):
      common.append(item[0])
  return common


def remove_hints(df, max_size=6, thresh=0.33):   
  '''Remove common shingles from titles.'''

  domains = df['domain'].value_counts().index.tolist()
  for domain in domains:
    domain_idx = df[df['domain'] == domain].index
    titles = df.ix[domain_idx, 'title'].copy(deep=True).tolist()
    for size in range(max_size+1)[::-1]:
      common = find_common_shingles(titles, size, thresh=thresh)
      if common:
        for c in common:
          if c.lower() != 'trump_propn':
            print('{},{},{}'.format(domain, len(titles), c))
            titles = [t.replace(c.strip(), '').strip() for t in titles]
            titles = [re.sub('\s+', ' ', t) for t in titles]
        for row, title in zip(domain_idx, titles):
          df.set_value(row, 'tokenized', title)
  return df


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
  '''Create vocabulary objects.'''

  c = Counter()
  for item in text_list:
    try:
      c.update(item.split())
    except:
      print('skipping "{}"'.format(item))
    
  print('dictionary size: {}'.format(len(c)))

  vocab_list = [(k, v) for k,v in c.iteritems()]
  vocab_list.sort(key=lambda x: -1 * x[1])

  vocab_word2idx = {x[0]: ix for ix,x in enumerate(vocab_list)}
  vocab_idx2word = {v: k for k,v in vocab_word2idx.iteritems()}

  return vocab_list, vocab_word2idx, vocab_idx2word


def write_dataset(output_file, df, keep_stops, min_sents, vocab, word2idx, idx2word):
  '''Helper to write dataset and configuration params to pickle.'''

  output_file = os.path.join(output_file)

  data = {
    'df': df,
    'keep_stops': keep_stops,
    'min_sents': min_sents,
    'vocab': vocab,
    'word2idx': word2idx,
    'idx2word': idx2word
  }

  with open(output_file, 'w') as f:
    _ = pickle.dump(data, f)
    print('wrote to {}'.format(output_file))


def read_dataset(filepath):
  '''Read and unpack the pre-processed data.'''

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


def dropout_tokens(docs, frac_drop=None, num_drop=None):
  '''Randomly drop out tokens from input sequences, either by fraction of total or
  by absolute count.
  '''

  def drop(indexes, num_drop):
    drop_idx = np.random.choice(indexes, this_num_drop, replace=False)
    _ = [indexes.remove(x) for x in drop_idx]
    return indexes

  if frac_drop:
    for doc_idx, doc in enumerate(docs):
      indexes = range(len(doc))
      this_num_drop = int(frac_drop * len(doc))
      indexes = drop(indexes, num_drop)
      docs[doc_idx] = [doc[x] for x in indexes]
    return docs

  if num_drop:
    for doc_idx, doc in enumerate(docs):
      indexes = range(len(doc))
      this_num_drop = min(len(indexes), num_drop)
      indexes = drop(indexes, num_drop)
      docs[doc_idx] = [doc[x] for x in indexes]
    return docs


def parse_model_name(name):
  '''Parse the model name to get parameter values used in training it.'''

  name = os.path.basename(name)
  args = name.split('.h5')[0].split('_')

  gru_dim, num_gru, embed_dim, dense_dim = map(int, args[:4])
  dropout = float(args[4])
  if args[5] == 'True':
    bidirectional = True
  else:
    bidirectional = False
  maxlen, topn = map(int, args[6:8])
  batch_size = int(args[8])
  output_size = int(args[9])
  gru_activation = args[10]

  return (gru_dim, num_gru, embed_dim, dense_dim, dropout, bidirectional, maxlen,
    topn, batch_size, output_size, gru_activation)

