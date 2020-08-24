import numpy as np
import logging
import gensim
import warnings
from gensim.models import Word2Vec
from gensim.models.word2vec import  LineSentence
import multiprocessing
import read_for_word2vec
import os
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

model = gensim.models.Word2Vec.load("1million.word2vec.model")
my_dict = {}
for idx, key in enumerate(model.wv.vocab):
    print('key {}'.format(key))
    my_dict[key] = model.wv[key]
    # Or my_dict[key] = model.wv.get_vector(key)
    # Or my_dict[key] = model.wv


arr = np.load('1million.word2vec.model.syn1neg.npy')
arr2 = np.load('1million.word2vec.model.wv.syn0.npy')
print(arr.shape)
print(arr2.shape)
'''
for i in range(10):
    print(arr[i,:5])
    print(arr2[i,:5])
'''