import numpy as np
import io
import os
import os
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
data_list = list()
word_dict = {}
dataset_x = []
dataset_y = []
'''
def strip_first_col(fname, delimiter=None):
    print('it is here')
    with open(fname, 'r') as fin:
        lines = fin.readlines()
        print(len(lines))
        for line in lines:
            print(line.split(delimiter,1)[0], line.split(delimiter,1)[1])
            list_of_words.append(line.split(delimiter,1)[0].encode())
            print(line.split(delimiter,1)[1])
            yield line.split(delimiter, 1)[1]
'''
cachedStopWords = stopwords.words("english")
nltk.download('stopwords')

def extract_data(training_files, training_labels, test_files, test_labels):
    file_count = 0
    count = 1
    training_x = []
    test_x = []
    for name in training_files:
        with open(name, 'r') as f:
            print('processing file', file_count)
            doc_vec = list()
            for line in f.readlines():
                for word in line.split():
                    if word not in word_dict:
                        word_dict[word] = count
                        count = count +1
                    if word not in cachedStopWords:
                        if count%500 == 0 and count >0:
                            doc_vec.append(0)
                        else:
                            doc_vec.append(word_dict[word])
        doc_arr = np.array(doc_vec)
        training_x.append(doc_arr)
        file_count = file_count +1
    for name in test_files:
        with open(name, 'r') as f:
            print('processing file', file_count)
            doc_vec = list()
            for line in f.readlines():
                for word in line.split():
                    if word not in word_dict and word not in cachedStopWords:
                        doc_vec.append(0)
                    elif word not in cachedStopWords:
                        doc_vec.append(word_dict[word])
        doc_arr = np.array(doc_vec)
        test_x.append(doc_arr)
        file_count = file_count +1

    return training_x, training_labels, test_x, test_labels, count

def create_dataset(paths):
    count = 0
    file_count = 0
    file_list = list()
    #global corpus, sentences
    length = list()
    full_length = list()
    for path in paths:
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                lent = 0
               # print('files {} '. format(os.path.join(root, name)))
                file_list.append(os.path.join(root, name))
                '''
                with open(os.path.join(root,name), 'r') as f:
                    print('processing file', file_count)
                    doc_vec = list()
                    for line in f.readlines():
                        for word in line.split():
                            if word not in word_dict:
                                word_dict[word] = count
                                count = count +1
                            if word not in cachedStopWords:
                                doc_vec.append(word_dict[word])
                doc_arr = np.array(doc_vec)
                dataset_x.append(doc_arr)
                '''
                if path == paths[0]:
                    dataset_y.append(0)
                else:
                    dataset_y.append(1)
                file_count = file_count+1
    return file_list, dataset_y


def create_train_test(a, b, x):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    ind = int(x*len(a))
    a = a[p]
    b = b[p]
    print(len(a), len(b))
    return a[:ind], b[:ind], a[ind+1:], b[ind+1:]
'''

filename = "word_vectors.vector"
with io.open(filename,'r',encoding='utf8') as f:
    text = f.read()
# process Unicode text
with io.open(filename,'w',encoding='utf8') as f:
    f.write(text)

list_of_words = list()
with open(filename, 'r') as fin:
    lines = fin.readlines()
    print(len(lines))
    for line in lines:
        #print(line.split(None, 1)[0], line.split(None, 1)[1])
        list_of_words.append(line.split(None, 1)[0].encode())
        num_str = line.split(None, 1)[1]
        nums = num_str.split()
        nums = [float(i) for i in nums]
        #print(len(nums))
        data_list.append(nums)

print(len(data_list))

data = np.zeros(shape=[len(data_list), 100])

#l = strip_first_col(filename)
for i in range(len(data_list)):
    #print(data_list[i])
    data[i,:] = np.array(data_list[i])


print(data.shape)
print(data[25,30])

paths = list()
paths.append('C:\\Users\\Administrator\\PycharmProjects\\BiLSTM\\Reliable')
paths.append('C:\\Users\\Administrator\\PycharmProjects\\BiLSTM\\Unreliable')

x, y = create_dataset(paths)
x = np.array(x)
y = np.array(y)
train_x, train_y, test_x, test_y = create_train_test(x, y, 0.7)
print(type(train_x))
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
max = 0
for i in range(len(train_x)):
    if max < len(train_x[i]):
        max = len(train_x[i])

print(max)


paths = list()
paths.append('C:\\Users\\Administrator\\PycharmProjects\\BiLSTM\\Reliable')
paths.append('C:\\Users\\Administrator\\PycharmProjects\\BiLSTM\\Unreliable')

file_list, y = create_dataset(paths)
file_list = np.array(file_list)
y = np.array(y)
training_files, training_label, test_files, test_labels = create_train_test(file_list, y, 0.7)
train_x, train_y, test_x, test_y,count = extract_data(training_files, training_label, test_files, test_labels)
train_x = np.array(train_x)
test_x = np.array(test_x)
print(type(train_x))
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
max = 0
for i in range(len(train_x)):
    if max < len(train_x[i]):
        max = len(train_x[i])

print(max)
'''