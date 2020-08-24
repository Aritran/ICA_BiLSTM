import logging
import gensim
import warnings
from gensim.models import Word2Vec
from gensim.models.word2vec import  LineSentence
import multiprocessing
import read_for_word2vec
import os
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

sentences = list()
corpora = ""

#def read_input2(input_file, sentences, corpora):

def read_input(input_file,sentences,corpora):
    """This method reads the input file which is in gzip format"""

    print("reading file {0}...this may take a while".format(input_file))
    with open(input_file, 'rb') as f:
        datalines = f.readlines()
        line = ""
        for data in datalines:
            try:
                data = str(data.decode('utf-8'))
                data_line = data.split('\t')
                if len(data_line)==2:
                    if data_line[0] == 'EOS':
                        sentences.append(line)
                        line = ""
                        corpora = corpora + '\n'
                    else:
                        line = line + " "+data_line[0]
                        corpora = corpora + str(data_line[0])
            except Exception as e:
                print('non ascii' + str(e))

    #for line in sentences:
        # if (i % 10000 == 0):
        #   logging.info("read {0} reviews".format(i))
        # do some pre-processing and return list of words for each review
        # text
     #   yield gensim.utils.simple_preprocess(line)

read_input('new_corpus_train.tsv',sentences,corpora)
read_input('new_corpus_test.tsv',sentences,corpora)
print(sentences)
new_sentences = LineSentence(corpora)
#print(new_sentences.max_sentence_length)
lines, model_out, vector_out = "words.txt", "word2vec.model", "word_vectors.vector"

model = Word2Vec([s.split() for s in sentences], sg=1, size=200, window=8, min_count=0, workers=multiprocessing.cpu_count())
    # ?????  ??????
for i in range(3000):
    model.train([s.split() for s in sentences], total_examples=len(sentences), epochs=model.iter)

    # trim unneeded model memory = use(much) less RAM
    # model.init_sims(replace=True)
print(model.most_similar('attack'))
model.save(model_out)
model.wv.save_word2vec_format(vector_out)

#model = Word2Vec(LineSentence(inp), size=400, window=5, min_count=5, workers=multiprocessing.cpu_count())