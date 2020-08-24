from __future__ import print_function
import numpy as np
import zois_dataset
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from sklearn.metrics import classification_report, accuracy_score
#import numpy as np
from keras.datasets import imdb
import COVID_dataextract
import time


max_features = 20000
# cut texts after this number of words
# (among top max_features most common words)
#maxlen = 500
maxlen = 20
#batch_size = 32
batch_size = 10
print('Loading data...')
paths = list()
paths.append('C:\\Users\\Administrator\\PycharmProjects\\BiLSTM\\Reliable')
paths.append('C:\\Users\\Administrator\\PycharmProjects\\BiLSTM\\Unreliable')

data_x, data_y, ind = COVID_dataextract.read_data()
data_x = np.array(data_x)
data_y = np.array(data_y)
print(len(data_x), len(data_y))
train_x, train_y, test_x, test_y = COVID_dataextract.create_train_test(data_x, data_y, 0.7)
x_train, y_train, x_test, y_test, max_features = COVID_dataextract.create_vec_data(train_x, train_y, test_x, test_y)
x_train = np.array(x_train)
x_test = np.array(x_test)
'''

file_list, y = zois_dataset.create_dataset(paths)
file_list = np.array(file_list)
y = np.array(y)
training_files, training_label, test_files, test_labels = zois_dataset.create_train_test(file_list, y, 0.7)
x_train, y_train, x_test, y_test, max_features = zois_dataset.extract_data(training_files, training_label, test_files, test_labels)
x_train = np.array(x_train)
x_test = np.array(x_test)
'''
print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
y_train = np.array(y_train)
y_test = np.array(y_test)

model = Sequential()
model.add(Embedding(max_features, 64, input_length=maxlen))
#model.add(Bidirectional(LSTM(64)))
model.add(Bidirectional(LSTM(32)))
#model.add(LSTM(64))
#model.add(LSTM(32))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

start_time = time.time()
print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=3,
          validation_data=[x_test, y_test])

score = model.evaluate(x_test, y_test, verbose=0)
#print('Test score:', score[0])
predicted_training = model.predict_classes(x_train)
print('Time for training {}'.format(time.time()-start_time))
print('Training classification score')
print('Training accuracy {}'.format(accuracy_score(y_train, predicted_training)))
print(classification_report(y_train, predicted_training, digits=3))
print('Number of parameters {}'.format(model.count_params()))
print(model.summary())
# evaluate model with sklearn
predicted_classes = model.predict_classes(x_test)
predicted_training = model.predict_classes(x_train)
#target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6', 'class 7', 'class 8', 'class 9']

#predicted_classes = np.argmax(predictions_last_epoch, axis=1)
print('Test classification score')
print('Test accuracy {}'.format(accuracy_score(y_test, predicted_classes)))
#print(classification_report(y_train, predicted_training, digits=3))
print(classification_report(y_test, predicted_classes, digits=3))






'''
#import keras
import os
from nltk.corpus import stopwords
import nltk
#nltk.download('stopwords')

from nltk.tokenize import word_tokenize
cachedStopWords = stopwords.words("english")
nltk.download('stopwords')
def create_dataset(paths):
    count = 0
    #global corpus, sentences
    length = list()
    full_length = list()
    for path in paths:
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                lent = 0
               # print('files {} '. format(os.path.join(root, name)))
                with open(os.path.join(root,name), 'r') as f:
                    print('processing file {}'.format(count))
                    contents = f.read()
                    text_tokens = word_tokenize(contents)
                    tokens_without_sw = [word for word in text_tokens if not word in cachedStopWords]
                    lent = len(tokens_without_sw)
                    full_length.append(len(contents.split()))
                    #print('lent {}'.format(lent))
                    length.append(lent)
                count = count + 1
    return max(length), max(full_length)

paths = list()
paths.append('C:\\Users\\Administrator\\PycharmProjects\\BiLSTM\\Reliable')
paths.append('C:\\Users\\Administrator\\PycharmProjects\\BiLSTM\\Unreliable')

print(create_dataset(paths))
'''

