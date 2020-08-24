

import os
yourpath = 'C:\\Users\\Administrator\\PycharmProjects\\BiLSTM\\Reliable'
corpus = ""
sentences = list()

def read_files(paths):
    count = 0
    global corpus, sentences
    for path in paths:
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
               # print('files {} '. format(os.path.join(root, name)))
                with open(os.path.join(root,name), 'r') as f:
                    print('processing file {}'.format(count))
                    datalines = f.readlines()
                    for line in datalines:
                        sentences_perline = line.split('.')
                        for line_sentence in sentences_perline:
                            sentences.append(line_sentence)
                            corpus = corpus + str(line_sentence) + '\n'
                count = count + 1
    return corpus, sentences



'''
read_files(yourpath)
yourpath = 'C:\\Users\\Administrator\\PycharmProjects\\BiLSTM\\Unreliable'
print('total number of sentences {}'.format(len(sentences)))
read_files(yourpath)

print('total number of sentences {}'.format(len(sentences)))
'''
